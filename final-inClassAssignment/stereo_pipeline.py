"""
final-inClassAssignment/stereo_pipeline.py
Uncalibrated stereo pipeline. Computes F, E, R/t, estimates object distance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class StereoResult:
    F: np.ndarray
    E: np.ndarray
    R: np.ndarray
    t_unit: np.ndarray
    t_scaled_m: np.ndarray
    estimated_distance_m: float
    object_point_3d_m: np.ndarray
    inlier_count: int
    match_count: int


def _resize_for_display(bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Return (display_image, scale) where original_px = display_px / scale."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr.copy(), 1.0
    scale = max_side / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def interactive_pick_one_point(
    bgr: np.ndarray,
    window_title: str,
    instruction: str,
    max_display_side: int = 1600,
) -> Tuple[float, float]:
    """
    Show image, wait for one left click (same pixel frame as bgr).
    Large images are shown scaled; click coordinates map back to full resolution.
    """
    display, scale = _resize_for_display(bgr, max_display_side)
    state: dict = {"pt_disp": None}

    def on_mouse(event: int, x: int, y: int, flags: int, param: Optional[np.ndarray]) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and state["pt_disp"] is None:
            state["pt_disp"] = (x, y)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, on_mouse)

    while state["pt_disp"] is None:
        vis = display.copy()
        h = vis.shape[0]
        cv2.putText(vis, instruction, (10, min(28, h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, "ESC cancel", (10, min(56, h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow(window_title, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Pick cancelled.")

    x_d, y_d = state["pt_disp"]
    ox = float(x_d) / scale
    oy = float(y_d) / scale

    vis = display.copy()
    cv2.circle(vis, (x_d, y_d), 8, (0, 0, 255), 2)
    cv2.putText(vis, "Space = continue", (10, min(28, vis.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(window_title, vis)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (32, 13, 10):  # space, enter
            break
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Pick cancelled.")

    cv2.destroyWindow(window_title)
    return ox, oy


def run_pick_points(left_path: Path, right_path: Path, max_display_side: int = 1600) -> None:
    """Open left then right image; print --object-left/right-* flags for the pipeline."""
    left_bgr = cv2.imread(str(left_path))
    right_bgr = cv2.imread(str(right_path))
    if left_bgr is None:
        raise RuntimeError(f"Could not read left image: {left_path}")
    if right_bgr is None:
        raise RuntimeError(f"Could not read right image: {right_path}")

    lx, ly = interactive_pick_one_point(
        left_bgr,
        "Stereo Pick — LEFT (same file as --left)",
        "Click the object point in the LEFT image",
        max_display_side=max_display_side,
    )
    rx, ry = interactive_pick_one_point(
        right_bgr,
        "Stereo Pick — RIGHT (same file as --right)",
        "Click the SAME 3D point in the RIGHT image",
        max_display_side=max_display_side,
    )
    cv2.destroyAllWindows()

    print("\n--- Copy into your pipeline command ---")
    print(
        f"  --object-left-x {lx:.2f} --object-left-y {ly:.2f} "
        f"--object-right-x {rx:.2f} --object-right-y {ry:.2f}"
    )
    print("\n--- PowerShell example (fill in other args) ---")
    print(
        f'python final-inClassAssignment/stereo_pipeline.py `\n'
        f'  --left "{left_path}" `\n'
        f'  --right "{right_path}" `\n'
        f'  --intrinsics final-inClassAssignment/camera_params.npz `\n'
        f'  --baseline-m <YOUR_BASELINE> `\n'
        f"  --object-left-x {lx:.2f} --object-left-y {ly:.2f} "
        f"--object-right-x {rx:.2f} --object-right-y {ry:.2f} `\n"
        f'  --ground-truth-m <OPTIONAL> `\n'
        f'  --output-dir final-inClassAssignment/results'
    )


def load_intrinsics(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path))
    if "K" in data:
        K = data["K"]
    elif "camera_matrix" in data:
        K = data["camera_matrix"]
    else:
        raise ValueError(f"Intrinsics file {npz_path} does not contain 'K' or 'camera_matrix'.")
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics matrix must be 3x3, got {K.shape}.")
    return K.astype(np.float64)


def detect_and_match(left_gray: np.ndarray, right_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)
    if des1 is None or des2 is None:
        raise RuntimeError("Could not compute ORB descriptors. Ensure images have texture and are valid.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 15:
        raise RuntimeError(f"Not enough good matches ({len(good)}). Capture richer texture/non-planar scene.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, good


def compute_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        raise RuntimeError("Failed to estimate fundamental matrix.")
    if F.shape != (3, 3):
        F = F[:3, :3]
    return F, mask.ravel().astype(bool)


def triangulate_object(
    K: np.ndarray,
    R: np.ndarray,
    t_scaled_m: np.ndarray,
    object_left_xy: np.ndarray,
    object_right_xy: np.ndarray,
) -> Tuple[np.ndarray, float]:
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t_scaled_m.reshape(3, 1)])

    x1 = np.array(object_left_xy, dtype=np.float64).reshape(2, 1)
    x2 = np.array(object_right_xy, dtype=np.float64).reshape(2, 1)

    X_h = cv2.triangulatePoints(P1, P2, x1, x2)
    if abs(X_h[3, 0]) < 1e-12:
        raise RuntimeError("Triangulation produced invalid homogeneous coordinate.")
    X = (X_h[:3, 0] / X_h[3, 0]).reshape(3)
    dist = float(np.linalg.norm(X))
    return X, dist


def draw_epipolar(left_bgr: np.ndarray, right_bgr: np.ndarray, F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]

    left = left_bgr.copy()
    right = right_bgr.copy()

    pt1 = np.array([[p1]], dtype=np.float32)
    pt2 = np.array([[p2]], dtype=np.float32)
    line_in_right = cv2.computeCorrespondEpilines(pt1, 1, F).reshape(-1, 3)[0]
    line_in_left = cv2.computeCorrespondEpilines(pt2, 2, F).reshape(-1, 3)[0]

    def draw_line(img: np.ndarray, line: np.ndarray, color: Tuple[int, int, int]) -> None:
        a, b, c = line
        if abs(b) > 1e-6:
            y0 = int((-c - a * 0) / b)
            y1 = int((-c - a * (img.shape[1] - 1)) / b)
            cv2.line(img, (0, y0), (img.shape[1] - 1, y1), color, 2)

    draw_line(right, line_in_right, (0, 255, 255))
    draw_line(left, line_in_left, (0, 255, 255))
    cv2.circle(left, tuple(np.round(p1).astype(int)), 8, (0, 0, 255), -1)
    cv2.circle(right, tuple(np.round(p2).astype(int)), 8, (0, 0, 255), -1)

    out_h = max(h1, h2)
    canvas = np.zeros((out_h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = left
    canvas[:h2, w1:w1 + w2] = right
    cv2.putText(canvas, "Left", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(canvas, "Right", (w1 + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return canvas


def draw_annotated_setup(setup_bgr: np.ndarray, estimated_m: float, ground_truth_m: Optional[float], click_xy: np.ndarray) -> np.ndarray:
    out = setup_bgr.copy()
    p = tuple(np.round(click_xy).astype(int))
    cv2.circle(out, p, 10, (0, 0, 255), -1)
    cv2.putText(out, "Object", (p[0] + 12, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"Estimated distance: {estimated_m:.3f} m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if ground_truth_m is not None:
        cv2.putText(out, f"Ground truth: {ground_truth_m:.3f} m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        err = abs(estimated_m - ground_truth_m)
        rel = (err / ground_truth_m * 100.0) if ground_truth_m > 1e-9 else 0.0
        cv2.putText(out, f"Absolute error: {err:.3f} m ({rel:.2f}%)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    return out


def compute_stereo(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    K: np.ndarray,
    baseline_m: float,
    object_left_xy: np.ndarray,
    object_right_xy: np.ndarray,
) -> StereoResult:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    pts1, pts2, good = detect_and_match(left_gray, right_gray)

    F, inlier_mask = compute_fundamental(pts1, pts2)
    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]
    if len(inlier_pts1) < 12:
        raise RuntimeError(f"Too few inliers after RANSAC ({len(inlier_pts1)}).")

    E = K.T @ F @ K
    _, R, t_unit, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K)
    t_unit = t_unit.reshape(3)
    norm_t = np.linalg.norm(t_unit)
    if norm_t < 1e-12:
        raise RuntimeError("Recovered translation direction has near-zero norm.")
    t_unit = t_unit / norm_t
    t_scaled_m = t_unit * float(baseline_m)

    obj_3d_m, est_distance_m = triangulate_object(K, R, t_scaled_m, object_left_xy, object_right_xy)

    return StereoResult(
        F=F,
        E=E,
        R=R,
        t_unit=t_unit,
        t_scaled_m=t_scaled_m,
        estimated_distance_m=est_distance_m,
        object_point_3d_m=obj_3d_m,
        inlier_count=int(len(inlier_pts1)),
        match_count=int(len(good)),
    )


def fmt_mat(name: str, arr: np.ndarray) -> str:
    return f"{name} =\n{np.array2string(arr, precision=6, suppress_small=False)}\n"


def latex_escape_plain(s: str) -> str:
    """Escape text for LaTeX running text (headers, title, author)."""
    t = str(s)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("_", r"\_"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ):
        t = t.replace(a, b)
    return t


def latex_escape_texttt(s: str) -> str:
    """Escape text for use inside LaTeX \\texttt{...} (paths, short labels)."""
    t = str(s).replace("\\", "/")
    for a, b in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("_", r"\_"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ):
        t = t.replace(a, b)
    return t


def latex_sanitize_verbatim(s: str) -> str:
    """Prevent accidentally closing a verbatim environment from pasted numeric output."""
    return str(s).replace("\\end{verbatim}", "\\end{verbatim} ")


def build_latex_report(
    template: str,
    *,
    left_path: Path,
    right_path: Path,
    baseline_m: float,
    object_left_xy: np.ndarray,
    object_right_xy: np.ndarray,
    ground_truth_m: Optional[float],
    result: StereoResult,
    abs_err: float,
    rel_err: float,
    report_student_name: str,
    report_course_code: str,
    report_course_title: str,
) -> str:
    gt_val = f"{ground_truth_m:.6f}" if ground_truth_m is not None else "N/A"
    f_block = latex_sanitize_verbatim(np.array2string(result.F, precision=6))
    e_block = latex_sanitize_verbatim(np.array2string(result.E, precision=6))
    r_block = latex_sanitize_verbatim(np.array2string(result.R, precision=6))
    t_unit_block = latex_sanitize_verbatim(np.array2string(result.t_unit, precision=6))
    t_scaled_block = latex_sanitize_verbatim(np.array2string(result.t_scaled_m, precision=6))

    xyz = np.asarray(result.object_point_3d_m).reshape(3)
    xyz_str = latex_escape_texttt(f"[{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}]")

    repl = {
        "__REPORT_STUDENT_NAME__": latex_escape_plain(report_student_name),
        "__REPORT_COURSE_CODE__": latex_escape_plain(report_course_code),
        "__REPORT_COURSE_TITLE__": latex_escape_plain(report_course_title),
        "__LEFT_IMAGE__": latex_escape_texttt(str(left_path)),
        "__RIGHT_IMAGE__": latex_escape_texttt(str(right_path)),
        "__BASELINE_M__": f"{baseline_m:.6f}",
        "__OBJECT_LEFT__": latex_escape_texttt(str(np.asarray(object_left_xy).tolist())),
        "__OBJECT_RIGHT__": latex_escape_texttt(str(np.asarray(object_right_xy).tolist())),
        "__GROUND_TRUTH_M__": latex_escape_texttt(gt_val),
        "__INLIER_COUNT__": str(int(result.inlier_count)),
        "__MATCH_COUNT__": str(int(result.match_count)),
        "__OBJECT_XYZ_M__": xyz_str,
        "__F_MATRIX__": f_block,
        "__E_MATRIX__": e_block,
        "__R_MATRIX__": r_block,
        "__T_UNIT__": t_unit_block,
        "__T_SCALED__": t_scaled_block,
        "__EST_DISTANCE_M__": f"{result.estimated_distance_m:.6f}",
        "__ABS_ERROR_M__": f"{abs_err:.6f}",
        "__REL_ERROR_PCT__": f"{rel_err:.3f}",
    }
    out = template
    for key, val in repl.items():
        out = out.replace(key, val)
    return out


def write_results(
    out_dir: Path,
    result: StereoResult,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    setup_bgr: np.ndarray,
    object_left_xy: np.ndarray,
    object_right_xy: np.ndarray,
    ground_truth_m: Optional[float],
    left_path: Path,
    right_path: Path,
    baseline_m: float,
    report_student_name: str,
    report_course_code: str,
    report_course_title: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    matrices_text = [
        fmt_mat("F", result.F),
        fmt_mat("E", result.E),
        fmt_mat("R", result.R),
        f"t_unit = {result.t_unit}\n",
        f"t_scaled_m = {result.t_scaled_m}\n",
        f"object_point_3d_m = {result.object_point_3d_m}\n",
        f"estimated_distance_m = {result.estimated_distance_m:.6f}\n",
        f"inlier_count = {result.inlier_count}\n",
        f"match_count = {result.match_count}\n",
    ]
    (out_dir / "matrices.txt").write_text("\n".join(matrices_text), encoding="utf-8")

    cv2.imwrite(str(out_dir / "stereo_left.jpg"), left_bgr)
    cv2.imwrite(str(out_dir / "stereo_right.jpg"), right_bgr)

    epi = draw_epipolar(left_bgr, right_bgr, result.F, object_left_xy, object_right_xy)
    cv2.imwrite(str(out_dir / "epipolar_lines.jpg"), epi)

    annotated = draw_annotated_setup(setup_bgr, result.estimated_distance_m, ground_truth_m, object_left_xy)
    cv2.imwrite(str(out_dir / "annotated_setup.jpg"), annotated)

    # Match preview for report
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1200)
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)
    preview = np.zeros((500, 800, 3), dtype=np.uint8)
    if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:60]
        preview = cv2.drawMatches(left_bgr, kp1, right_bgr, kp2, matches, None, flags=2)
    cv2.imwrite(str(out_dir / "matches_inliers.jpg"), preview)

    report_template_path = Path(__file__).with_name("report_template.tex")
    template = report_template_path.read_text(encoding="utf-8")
    abs_err = abs(result.estimated_distance_m - ground_truth_m) if ground_truth_m is not None else 0.0
    rel_err = (abs_err / ground_truth_m * 100.0) if (ground_truth_m is not None and ground_truth_m > 1e-9) else 0.0

    report_tex = build_latex_report(
        template,
        left_path=left_path,
        right_path=right_path,
        baseline_m=baseline_m,
        object_left_xy=object_left_xy,
        object_right_xy=object_right_xy,
        ground_truth_m=ground_truth_m,
        result=result,
        abs_err=abs_err,
        rel_err=rel_err,
        report_student_name=report_student_name,
        report_course_code=report_course_code,
        report_course_title=report_course_title,
    )
    (out_dir / "report.tex").write_text(report_tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncalibrated stereo assignment pipeline")
    p.add_argument("--left", type=str, default="", help="Path to left image")
    p.add_argument("--right", type=str, default="", help="Path to right image")
    p.add_argument("--intrinsics", type=str, default="", help="Path to npz file containing K/camera_matrix")
    p.add_argument("--baseline-m", type=float, default=0.0, help="Measured baseline in meters")
    p.add_argument("--object-left-x", type=float, default=np.nan)
    p.add_argument("--object-left-y", type=float, default=np.nan)
    p.add_argument("--object-right-x", type=float, default=np.nan)
    p.add_argument("--object-right-y", type=float, default=np.nan)
    p.add_argument("--ground-truth-m", type=float, default=np.nan, help="Ground truth object distance in meters")
    p.add_argument("--setup-image", type=str, default="", help="Photo of camera setup for annotation")
    p.add_argument("--output-dir", type=str, default="final-inClassAssignment/results")
    p.add_argument("--dry-run", action="store_true", help="Create output folder and template placeholders only")
    p.add_argument(
        "--pick-points",
        action="store_true",
        help="Open left/right images to click the same object point; prints --object-* flags (needs --left and --right only)",
    )
    p.add_argument(
        "--pick-max-side",
        type=int,
        default=1600,
        help="Max width/height for on-screen display when picking points (full-res coords are still saved)",
    )
    p.add_argument(
        "--report-student-name",
        type=str,
        default="YOUR NAME",
        help="Student name for LaTeX report header and title page",
    )
    p.add_argument(
        "--report-course-code",
        type=str,
        default="CSC8830",
        help="Course code for LaTeX report header",
    )
    p.add_argument(
        "--report-course-title",
        type=str,
        default="Computer Vision",
        help="Course title for LaTeX report header",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README_dry_run.txt").write_text(
            "Dry run successful. Provide --left --right --intrinsics --baseline-m and object points to run full pipeline.\n",
            encoding="utf-8",
        )
        print(f"Dry run complete. Output directory initialized at: {out_dir}")
        return

    if args.pick_points:
        if not args.left or not args.right:
            raise ValueError("--pick-points requires --left and --right image paths.")
        run_pick_points(Path(args.left), Path(args.right), max_display_side=int(args.pick_max_side))
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_m <= 0:
        raise ValueError("Baseline must be > 0 meters.")
    if args.baseline_m < 0.01:
        print("Warning: very small baseline may cause unstable depth estimates.")

    left_path = Path(args.left)
    right_path = Path(args.right)
    intrinsics_path = Path(args.intrinsics)
    if not left_path.exists() or not right_path.exists() or not intrinsics_path.exists():
        raise FileNotFoundError("Input files missing. Check --left, --right, and --intrinsics.")

    left_bgr = cv2.imread(str(left_path))
    right_bgr = cv2.imread(str(right_path))
    if left_bgr is None or right_bgr is None:
        raise RuntimeError("Could not read input image(s).")

    K = load_intrinsics(intrinsics_path)
    obj_left = np.array([args.object_left_x, args.object_left_y], dtype=np.float64)
    obj_right = np.array([args.object_right_x, args.object_right_y], dtype=np.float64)
    if np.isnan(obj_left).any() or np.isnan(obj_right).any():
        raise ValueError("Object click coordinates are required (left and right x/y).")

    result = compute_stereo(left_bgr, right_bgr, K, args.baseline_m, obj_left, obj_right)

    setup_bgr = left_bgr
    if args.setup_image:
        maybe_setup = cv2.imread(args.setup_image)
        if maybe_setup is not None:
            setup_bgr = maybe_setup

    gt = None if np.isnan(args.ground_truth_m) else float(args.ground_truth_m)
    write_results(
        out_dir=out_dir,
        result=result,
        left_bgr=left_bgr,
        right_bgr=right_bgr,
        setup_bgr=setup_bgr,
        object_left_xy=obj_left,
        object_right_xy=obj_right,
        ground_truth_m=gt,
        left_path=left_path,
        right_path=right_path,
        baseline_m=float(args.baseline_m),
        report_student_name=args.report_student_name,
        report_course_code=args.report_course_code,
        report_course_title=args.report_course_title,
    )

    print("Pipeline completed successfully.")
    print(f"Estimated object distance: {result.estimated_distance_m:.4f} m")
    print(f"Inliers/matches: {result.inlier_count}/{result.match_count}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()

