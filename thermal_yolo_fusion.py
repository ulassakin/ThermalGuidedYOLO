import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


@dataclass
class Detection: # Every bounding box is stored in this format
    box: List[int]             # [x1, y1, x2, y2]
    confidence: float
    cls: int
    original_box: List[int]
    refined_by_thermal: bool = False
    source: str = "yolo"       # "yolo" or "thermal_only"
    match_score: float = 0.0   # for debugging


@dataclass
class ThermalBlob:
    box: List[int]             # [x1, y1, x2, y2]
    area: int
    centroid: Tuple[float, float]
    mean_val: float
    max_val: float
    hot_fraction: float        # fraction of pixels in blob above threshold
    id: int


def clamp_box(box: List[int], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, W - 1)))
    y1 = int(max(0, min(y1, H - 1)))
    x2 = int(max(0, min(x2, W)))
    y2 = int(max(0, min(y2, H)))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return [x1, y1, x2, y2]


def box_area(box: List[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(b1: List[int], b2: List[int]) -> int:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(b1: List[int], b2: List[int]) -> float:
    inter = intersection_area(b1, b2)
    if inter <= 0:
        return 0.0
    union = box_area(b1) + box_area(b2) - inter
    return float(inter) / float(union) if union > 0 else 0.0


def expand_box(box: List[int], margin: float, W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = int(round(bw * margin))
    dy = int(round(bh * margin))
    return clamp_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], W, H)


class ThermalGuidedYOLORefiner:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        backend: str = "auto",  # "auto" | "ultralytics" | "yolov5hub"
        # Thermal mask params
        temp_percentile: float = 95.0,
        min_hot_area: int = 80,
        morph_kernel: int = 3,
        morph_iters: int = 1,
        # Matching / refinement
        min_overlap_ratio: float = 0.02,
        centroid_bonus: float = 0.25,      # boosts score if blob centroid inside YOLO box
        overlap_weight: float = 1.0,
        iou_weight: float = 0.3,
        match_score_threshold: float = 0.05,
        refine_margin: float = 0.20,       # allow expansion around YOLO box to catch missed parts
        refine_padding: int = 4,
        min_refined_size: int = 10,
        # Thermal-only gate
        add_thermal_only: bool = True,
        thermal_only_min_area: int = 120,
        thermal_only_max_area_frac: float = 0.25,  # max blob area relative to image area
        thermal_only_min_hot_fraction: float = 0.20,
        thermal_only_conf_base: float = 0.20,
        thermal_only_conf_gain: float = 0.70,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.backend = backend

        self.temp_percentile = temp_percentile
        self.min_hot_area = min_hot_area
        self.morph_kernel = morph_kernel
        self.morph_iters = morph_iters

        self.min_overlap_ratio = min_overlap_ratio
        self.centroid_bonus = centroid_bonus
        self.overlap_weight = overlap_weight
        self.iou_weight = iou_weight
        self.match_score_threshold = match_score_threshold

        self.refine_margin = refine_margin
        self.refine_padding = refine_padding
        self.min_refined_size = min_refined_size

        self.add_thermal_only = add_thermal_only
        self.thermal_only_min_area = thermal_only_min_area
        self.thermal_only_max_area_frac = thermal_only_max_area_frac
        self.thermal_only_min_hot_fraction = thermal_only_min_hot_fraction
        self.thermal_only_conf_base = thermal_only_conf_base
        self.thermal_only_conf_gain = thermal_only_conf_gain

        self._load_model()

        print(f"✅ Model: {self.model_path}")
        print(f"   Backend: {self._backend_name}")
        print(f"   YOLO conf threshold: {self.conf_threshold}")
        print(f"   Thermal percentile: {self.temp_percentile} (top {100 - self.temp_percentile:.1f}% hottest)")

    def _load_model(self):
        # Decide backend
        chosen = self.backend.lower()
        self._backend_name = "unknown"
        self._ultra = None
        self._y5 = None

        if chosen not in ("auto", "ultralytics", "yolov5hub"):
            raise ValueError("backend must be one of: auto, ultralytics, yolov5hub")

        if chosen in ("auto", "ultralytics"):
            try:
                from ultralytics import YOLO  # type: ignore
                self._ultra = YOLO(self.model_path)
                self._backend_name = "ultralytics"
                return
            except Exception:
                if chosen == "ultralytics":
                    raise

        # Fallback: YOLOv5 torch.hub
        import torch
        self._y5 = torch.hub.load("ultralytics/yolov5", "custom", path=self.model_path, force_reload=False)
        self._y5.conf = self.conf_threshold
        self._backend_name = "yolov5hub"

    def load_images(self, rgb_path: str, thermal_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        #We are loading both RGB and thermal image of the same frame
        # RGB
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # Thermal (keep raw)
        therm = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
        if therm is None:
            raise FileNotFoundError(f"Could not read thermal image: {thermal_path}")

        # Convert thermal to grayscale if needed
        if therm.ndim == 3:
            therm_gray = cv2.cvtColor(therm, cv2.COLOR_BGR2GRAY)
        else:
            therm_gray = therm

        # Resize thermal to match RGB if needed
        if rgb.shape[:2] != therm_gray.shape[:2]:
            print("⚠️  Thermal size != RGB size. Resizing thermal to match RGB (baseline alignment).")
            therm_gray = cv2.resize(therm_gray, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        # thermal_float for scoring (handle 16-bit etc.)
        therm_float = therm_gray.astype(np.float32)

        return rgb, therm_gray, therm_float

    def run_yolo(self, rgb: np.ndarray) -> List[Detection]:
        dets: List[Detection] = []
        #We are creating the prediction bounding boxes
        if self._backend_name == "ultralytics":
            # ultralytics YOLO expects BGR typically, but accepts numpy RGB in many builds;
            # to be safe: convert to BGR for inference.
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            results = self._ultra.predict(rgb_bgr, conf=self.conf_threshold, verbose=False)
            if not results:
                return dets
            r0 = results[0]
            if r0.boxes is None or len(r0.boxes) == 0:
                return dets
            xyxy = r0.boxes.xyxy.cpu().numpy()
            conf = r0.boxes.conf.cpu().numpy()
            cls = r0.boxes.cls.cpu().numpy()
            for b, c, k in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
                dets.append(Detection(
                    box=[x1, y1, x2, y2],
                    confidence=float(c),
                    cls=int(k),
                    original_box=[x1, y1, x2, y2],
                    refined_by_thermal=False,
                    source="yolo",
                ))
            return dets

        # yolov5 hub
        results = self._y5(rgb)  # RGB ok for yolov5 hub
        if len(results.xyxy[0]) == 0:
            return dets
        arr = results.xyxy[0].cpu().numpy()
        for *box, conf, cls in arr:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            dets.append(Detection(
                box=[x1, y1, x2, y2],
                confidence=float(conf),
                cls=int(cls),
                original_box=[x1, y1, x2, y2],
                refined_by_thermal=False,
                source="yolo",
            ))
        return dets

    def build_thermal_mask(self, therm_gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns:
          mask (uint8 0/255), threshold used (float)
        """
        # Use only non-zero if present; if all zeros -> empty
        nonzero = therm_gray[therm_gray > 0] #Take only nonzero pixels
        if nonzero.size == 0:
            return np.zeros_like(therm_gray, dtype=np.uint8), float("inf")

        thr = float(np.percentile(nonzero, self.temp_percentile))#Apply the percentile
        _, mask = cv2.threshold(therm_gray, thr, 255, cv2.THRESH_BINARY)
        #Creating the hot mask based on the precentile
        mask = mask.astype(np.uint8)

        # Morphology cleanup
        k = max(1, int(self.morph_kernel))
        kernel = np.ones((k, k), np.uint8)
        for _ in range(max(1, int(self.morph_iters))):
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Filter by area using contours -> redraw valid mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_hot_area:
                cv2.drawContours(valid, [cnt], -1, 255, -1)
        #Taking only the areas that have area mor than min_hot_area
        return valid, thr

    def extract_blobs(self, hot_mask: np.ndarray, therm_float: np.ndarray, thr: float) -> List[ThermalBlob]:
        """
        Build blob list from hot_mask using connected components.
        """
        #We are using both binary mask and thermal information here. 
        blobs: List[ThermalBlob] = []
        if hot_mask.max() == 0:
            return blobs

        num, labels, stats, centroids = cv2.connectedComponentsWithStats((hot_mask > 0).astype(np.uint8), connectivity=8)

        H, W = hot_mask.shape
        for i in range(1, num):  # 0 is background
            x, y, w, h, area = stats[i]
            if area < self.min_hot_area:
                continue

            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)#Creating the bounding box
            cx, cy = centroids[i]
            roi_mask = (labels[y1:y2, x1:x2] == i)
            roi_vals = therm_float[y1:y2, x1:x2][roi_mask]

            if roi_vals.size == 0:
                continue

            mean_val = float(np.mean(roi_vals))
            max_val = float(np.max(roi_vals))
            hot_fraction = float(np.mean(roi_vals >= thr)) if np.isfinite(thr) else 0.0

            blobs.append(ThermalBlob(
                box=[x1, y1, x2, y2],
                area=int(area),
                centroid=(float(cx), float(cy)),
                mean_val=mean_val,
                max_val=max_val,
                hot_fraction=hot_fraction,
                id=i
            ))

        return blobs

    def _overlap_ratio(self, box: List[int], hot_mask: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        roi = hot_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        hot = int(np.count_nonzero(roi))
        return float(hot) / float(roi.size)

    def _centroid_inside(self, blob: ThermalBlob, box: List[int]) -> bool:
        cx, cy = blob.centroid
        return (box[0] <= cx <= box[2]) and (box[1] <= cy <= box[3])

    def match_yolo_to_blobs(self, dets: List[Detection], blobs: List[ThermalBlob], hot_mask: np.ndarray) -> Dict[int, int]:
        """
        Greedy one-to-one assignment by match score.

        Returns:
          mapping: det_index -> blob_index
        """
        if not dets or not blobs:
            return {}

        pairs = []
        for di, d in enumerate(dets):
            # quick rejection using bbox intersection with blob bbox (cheap)
            for bi, b in enumerate(blobs):
                inter = intersection_area(d.box, b.box)
                if inter <= 0:
                    continue

                # Score components
                ov = self._overlap_ratio(d.box, hot_mask)  # thermal pixels inside YOLO box
                j = iou(d.box, b.box)
                bonus = self.centroid_bonus if self._centroid_inside(b, d.box) else 0.0

                score = self.overlap_weight * ov + self.iou_weight * j + bonus
                pairs.append((score, di, bi))

        pairs.sort(reverse=True, key=lambda x: x[0])

        assigned_d = set()
        assigned_b = set()
        mapping: Dict[int, int] = {}

        for score, di, bi in pairs:
            if score < self.match_score_threshold:
                break
            if di in assigned_d or bi in assigned_b:
                continue
            mapping[di] = bi
            assigned_d.add(di)
            assigned_b.add(bi)
            dets[di].match_score = float(score)

        return mapping

    def refine_box(self, det: Detection, hot_mask: np.ndarray, W: int, H: int) -> Tuple[List[int], bool]:
        """
        Refine YOLO box by searching in an expanded region and tightening to hot pixels.
        Allows expansion (important).
        """
        # Expand search region around YOLO box
        search = expand_box(det.box, self.refine_margin, W, H)
        sx1, sy1, sx2, sy2 = search
        roi = hot_mask[sy1:sy2, sx1:sx2]
        if roi.size == 0 or np.count_nonzero(roi) == 0:
            return det.box, False

        ys, xs = np.where(roi > 0)
        if ys.size == 0:
            return det.box, False

        # Tight bbox in global coords
        x1 = int(xs.min() + sx1)
        x2 = int(xs.max() + sx1)
        y1 = int(ys.min() + sy1)
        y2 = int(ys.max() + sy1)

        # Padding
        pad = int(self.refine_padding)
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad

        refined = clamp_box([x1, y1, x2, y2], W, H)

        # Size check
        if (refined[2] - refined[0]) < self.min_refined_size or (refined[3] - refined[1]) < self.min_refined_size:
            return det.box, False

        # Overlap ratio check in refined box to avoid junk
        ov = self._overlap_ratio(refined, hot_mask)
        if ov < self.min_overlap_ratio:
            return det.box, False

        return refined, True

    def thermal_only_conf(self, blob: ThermalBlob, thr: float) -> float:
        """
        Compute a conservative confidence score for thermal-only detections.
        """
        # hot_fraction already reflects how "strong" this blob is relative to thr
        strength = np.clip(blob.hot_fraction, 0.0, 1.0)
        conf = self.thermal_only_conf_base + self.thermal_only_conf_gain * strength
        return float(np.clip(conf, 0.0, 0.99))

    def add_thermal_only_dets(self, blobs: List[ThermalBlob], used_blob_idx: set, W: int, H: int, thr: float) -> List[Detection]:
        """
        Add unmatched blobs as thermal-only detections with safety filters.
        """
        if not self.add_thermal_only:
            return []

        out: List[Detection] = []
        img_area = float(W * H)
        max_area = int(self.thermal_only_max_area_frac * img_area)

        for bi, b in enumerate(blobs):
            if bi in used_blob_idx:
                continue

            if b.area < self.thermal_only_min_area:
                continue
            if b.area > max_area:
                continue
            if b.hot_fraction < self.thermal_only_min_hot_fraction:
                continue

            box = clamp_box(b.box, W, H)
            conf = self.thermal_only_conf(b, thr)

            out.append(Detection(
                box=box,
                confidence=conf,
                cls=0,  # assume 0=fire; adjust for your dataset
                original_box=box.copy(),
                refined_by_thermal=True,
                source="thermal_only",
                match_score=0.0
            ))
        return out

    def process(
        self,
        rgb_path: str,
        thermal_path: str,
        visualize: bool = True,
        save_viz: Optional[str] = None
    ) -> List[Detection]:

        print("\n" + "=" * 70)
        print("PROCESSING")
        print("=" * 70)
        print(f"RGB    : {rgb_path}")
        print(f"Thermal: {thermal_path}")

        rgb, therm_gray, therm_float = self.load_images(rgb_path, thermal_path)
        H, W = rgb.shape[:2]

        hot_mask, thr = self.build_thermal_mask(therm_gray)
        blobs = self.extract_blobs(hot_mask, therm_float, thr)

        print(f"Thermal: threshold={thr:.2f} | hot_blobs={len(blobs)}")

        dets = self.run_yolo(rgb)
        print(f"YOLO   : detections={len(dets)}")

        # Match
        mapping = self.match_yolo_to_blobs(dets, blobs, hot_mask)
        used_blobs = set(mapping.values())

        # Refine matched detections using GLOBAL hot_mask ROI (no contour clipping bugs)
        refined_count = 0
        for di, det in enumerate(dets):
            if di not in mapping:
                det.refined_by_thermal = False
                continue

            refined_box, ok = self.refine_box(det, hot_mask, W, H)
            if ok:
                det.box = refined_box
                det.refined_by_thermal = True
                refined_count += 1
            else:
                det.refined_by_thermal = False

        # Add thermal-only detections (optional, gated)
        thermal_only = self.add_thermal_only_dets(blobs, used_blobs, W, H, thr)
        final = dets + thermal_only

        def union_merge(dets, cls_id=0, iou_thr=0.05):
            pool = [d for d in dets if d.source == "thermal_only" and d.cls == cls_id]
            others = [d for d in dets if not (d.source == "thermal_only" and d.cls == cls_id)]
        
            merged_out = []
            while pool:
                a = pool.pop(0)
                group = [a]
        
                changed = True
                while changed:
                    changed = False
                    new_pool = []
                    for b in pool:
                        # group'taki herhangi biriyle çakışıyorsa birleştir
                        if any(iou(g.box, b.box) > iou_thr for g in group):
                            group.append(b)
                            changed = True
                        else:
                            new_pool.append(b)
                    pool = new_pool
        
                if len(group) > 1:
                    x1 = min(g.box[0] for g in group)
                    y1 = min(g.box[1] for g in group)
                    x2 = max(g.box[2] for g in group)
                    y2 = max(g.box[3] for g in group)
                    a.box = [x1, y1, x2, y2]
                    a.confidence = max(g.confidence for g in group)
        
                merged_out.append(a)
        
            return others + merged_out
        
        # final oluşturduktan sonra:
        final = union_merge(final, cls_id=0, iou_thr=0.05)

        self.print_summary(final, refined_count, len(thermal_only))

        if visualize:
            self.visualize(rgb, therm_gray, hot_mask, final, save_viz)

        return final

    def print_summary(self, dets: List[Detection], refined_count: int, thermal_only_count: int):
        yolo_total = sum(1 for d in dets if d.source == "yolo")
        yolo_unchanged = sum(1 for d in dets if d.source == "yolo" and not d.refined_by_thermal)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total detections         : {len(dets)}")
        print(f"YOLO detections          : {yolo_total}")
        print(f"  - refined by thermal   : {refined_count}")
        print(f"  - unchanged            : {yolo_unchanged}")
        print(f"Thermal-only added       : {thermal_only_count}")
        print("=" * 70)

    def visualize(self, rgb: np.ndarray, therm_gray: np.ndarray, hot_mask: np.ndarray, dets: List[Detection], save_viz: Optional[str]):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]

        ax1.imshow(rgb)
        ax1.set_title("RGB: Original YOLO Boxes", fontweight="bold")

        ax2.imshow(therm_gray, cmap="hot")
        ax2.imshow(hot_mask, alpha=0.35, cmap="Blues")
        ax2.set_title("Thermal + Hot Mask", fontweight="bold")

        ax3.imshow(rgb)
        ax3.set_title("RGB: Final Boxes (Refined + Thermal-only)", fontweight="bold")

        ax4.imshow(rgb)
        ax4.set_title("Comparison: Original (dashed) vs Final (solid)", fontweight="bold")

        colors = {
            "original": "yellow",
            "refined": "lime",
            "unchanged": "orange",
            "thermal_only": "cyan",
        }

        # Draw
        for d in dets:
            ox1, oy1, ox2, oy2 = d.original_box
            x1, y1, x2, y2 = d.box

            # ax1: original yolo only
            if d.source == "yolo":
                rect = plt.Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1,
                                     linewidth=2, edgecolor=colors["original"], facecolor="none")
                ax1.add_patch(rect)
                ax1.text(ox1, max(0, oy1 - 5), f"{d.confidence:.2f}",
                         color=colors["original"], fontsize=8, fontweight="bold")

            # ax3: final
            if d.source == "thermal_only":
                color = colors["thermal_only"]
            else:
                color = colors["refined"] if d.refined_by_thermal else colors["unchanged"]

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor="none")
            ax3.add_patch(rect)
            ax3.text(x1, max(0, y1 - 5), f"{d.confidence:.2f}",
                     color=color, fontsize=8, fontweight="bold")

            # ax4: comparison
            if d.source == "yolo":
                rect_o = plt.Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1,
                                       linewidth=2, edgecolor=colors["original"], linestyle="--", facecolor="none")
                ax4.add_patch(rect_o)

            rect_f = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor=color, facecolor="none")
            ax4.add_patch(rect_f)

        # Legends
        handles = [
            plt.Line2D([0], [0], color=colors["original"], lw=2, linestyle="--", label="Original YOLO"),
            plt.Line2D([0], [0], color=colors["refined"], lw=2, label="Thermal-refined"),
            plt.Line2D([0], [0], color=colors["unchanged"], lw=2, label="YOLO unchanged"),
            plt.Line2D([0], [0], color=colors["thermal_only"], lw=2, label="Thermal-only"),
        ]
        ax3.legend(handles=handles, loc="upper right", fontsize=9)
        ax4.legend(handles=handles, loc="upper right", fontsize=9)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis("off")

        plt.tight_layout()
        if save_viz:
            plt.savefig(save_viz, dpi=160, bbox_inches="tight")
            print(f"✅ Saved visualization: {save_viz}")
        plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to YOLO .pt model")
    p.add_argument("--rgb", required=True, help="Path to RGB image")
    p.add_argument("--thermal", required=True, help="Path to thermal image")
    p.add_argument("--backend", default="auto", choices=["auto", "ultralytics", "yolov5hub"])
    p.add_argument("--conf", type=float, default=0.25)

    p.add_argument("--temp_percentile", type=float, default=95.0)
    p.add_argument("--min_hot_area", type=int, default=80)

    p.add_argument("--min_overlap_ratio", type=float, default=0.02)
    p.add_argument("--match_score_threshold", type=float, default=0.05)
    p.add_argument("--refine_margin", type=float, default=0.20)
    p.add_argument("--refine_padding", type=int, default=4)

    p.add_argument("--no_thermal_only", action="store_true")
    p.add_argument("--save_viz", default=None)
    p.add_argument("--no_viz", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    refiner = ThermalGuidedYOLORefiner(
        model_path=args.model,
        conf_threshold=args.conf,
        backend=args.backend,

        temp_percentile=args.temp_percentile,
        min_hot_area=args.min_hot_area,

        min_overlap_ratio=args.min_overlap_ratio,
        match_score_threshold=args.match_score_threshold,
        refine_margin=args.refine_margin,
        refine_padding=args.refine_padding,

        add_thermal_only=(not args.no_thermal_only),
    )

    final = refiner.process(
        rgb_path=args.rgb,
        thermal_path=args.thermal,
        visualize=(not args.no_viz),
        save_viz=args.save_viz
    )

    print("\nFinal detections:")
    for i, d in enumerate(final, 1):
        print(f"{i:02d} | src={d.source:11s} refined={str(d.refined_by_thermal):5s} "
              f"conf={d.confidence:.3f} cls={d.cls} box={d.box} score={d.match_score:.3f}")


if __name__ == "__main__":
    main()
