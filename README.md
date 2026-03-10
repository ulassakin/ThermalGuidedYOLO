# ThermalGuidedYOLO

ThermalGuidedYOLO is a lightweight RGB–thermal fusion pipeline that refines YOLO detections using thermal hotspot information.  
It improves bounding box localization and can optionally add thermal-only detections without retraining a multimodal model.

---

## Example Result

![Example Result](Examples/result.png)

---

## Key Idea

RGB detectors can sometimes produce loose or fragmented detections or miss hot regions entirely.  
Thermal imagery provides strong evidence of high-temperature areas.

This pipeline uses thermal information at **inference time** to refine YOLO detections and improve localization.

---

## Pipeline

1. Run **YOLO** on the RGB image to generate candidate detections  
2. Extract **hot regions** from the thermal image using dynamic thresholding  
3. Perform **blob analysis** to identify thermal hotspots  
4. **Match detections with thermal blobs** using overlap and centroid checks  
5. **Refine bounding boxes** using the thermal mask  
6. Optionally add **thermal-only detections** for missed hotspots

---

## Features

- RGB + Thermal detection refinement
- Dynamic thermal hotspot extraction
- Blob-based hotspot analysis
- Bounding box tightening or expansion
- Optional thermal-only detections
- Visualization for debugging and comparison

---

## Usage

```bash
python thermal_guided_refine.py \
  --model best.pt \
  --rgb path/to/rgb.jpg \
  --thermal path/to/thermal.png
