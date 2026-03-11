# ThermalGuidedYOLO

ThermalGuidedYOLO is a lightweight RGB–thermal fusion pipeline designed for **fire and smoke detection**.  
It refines YOLO detections using thermal hotspot information to improve localization of **fire and smoke regions**, and can optionally add thermal-only detections without retraining a multimodal model.

The method assumes **synchronized RGB and thermal images captured from a dual-camera setup**.

---

## Example Result

![Example Result](Examples/result.png)

---

## Key Idea

RGB-based fire and smoke detectors can sometimes produce loose detections,
fragment smoke regions, or miss early fire hotspots.

Thermal imagery provides strong evidence of high-temperature regions,
which can help localize **active fire areas** and refine the bounding boxes
predicted by RGB detectors.

By combining RGB detection with thermal hotspot information,
the system improves **fire localization and detection reliability**.

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

- Fire and smoke detection refinement using RGB + thermal fusion
- Thermal hotspot extraction for fire localization
- Blob-based thermal hotspot analysis
- Bounding box tightening or expansion using thermal masks
- Optional thermal-only detections for missed fire hotspots
- Visualization for debugging fire/smoke detections

---

## Usage

```bash
python thermal_guided_refine.py \
  --model best.pt \
  --rgb path/to/rgb.jpg \
  --thermal path/to/thermal.png
```
---

## Applications

This approach can be useful in several fire monitoring scenarios where RGB and thermal cameras are available:

- **Wildfire detection systems**
- **Early fire hotspot detection**
- **UAV-based fire monitoring**
- **RGB–thermal surveillance systems**
- **Forest and industrial fire monitoring**
