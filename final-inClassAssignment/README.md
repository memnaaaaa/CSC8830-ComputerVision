# Uncalibrated Stereo (Single Camera, Two Positions)

This folder contains a pipeline for the uncalibrated stereo assignment.

## What this does

- Estimates Fundamental Matrix `F` from feature matches
- Computes Essential Matrix `E = K^T F K` (if intrinsics are provided)
- Recovers relative pose (`R`, `t` direction) and scales `t` using measured baseline
- Triangulates an object point and estimates camera-to-object distance
- Saves report-ready outputs (matrices, annotated image, epipolar image, LaTeX `report.tex`)

## Recommended workflow

1. Calibrate the camera (reuse `module2/calibrate_camera.py`) and save `K`.
2. Capture two images by translating camera sideways while preserving orientation.
3. Measure:
   - baseline in meters (camera center translation approximation)
   - ground-truth distance to chosen object (tape measure)
4. Run:

```bash
python final-inClassAssignment/stereo_pipeline.py --dry-run
```

To interactively get object pixel coordinates (opens two image windows; click the same 3D point; press Space after each click):

```bash
python final-inClassAssignment/stereo_pipeline.py --pick-points ^
  --left path/to/left.jpg ^
  --right path/to/right.jpg
```

Then run with real inputs:

```bash
python final-inClassAssignment/stereo_pipeline.py ^
  --left path/to/left.jpg ^
  --right path/to/right.jpg ^
  --intrinsics path/to/camera_params.npz ^
  --baseline-m 0.12 ^
  --object-left-x 730 --object-left-y 410 ^
  --object-right-x 694 --object-right-y 412 ^
  --ground-truth-m 1.65 ^
  --setup-image path/to/setup.jpg ^
  --report-student-name "Ada Lovelace" ^
  --report-course-code "CSC8830" ^
  --report-course-title "Computer Vision" ^
  --output-dir final-inClassAssignment/results
```

## Outputs

`results/` contains:
- `matrices.txt`
- `stereo_left.jpg`, `stereo_right.jpg` (copies of the stereo pair for the report)
- `annotated_setup.jpg`
- `epipolar_lines.jpg`
- `report.tex`
- `matches_inliers.jpg`

