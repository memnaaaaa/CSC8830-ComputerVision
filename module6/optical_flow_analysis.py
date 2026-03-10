"""
module6/optical_flow_analysis.py
Part A: Optical Flow Computation and Visualization
Computes dense optical flow using Farneback method and visualizes as video.
"""

import cv2
import numpy as np
import os


def compute_optical_flow_video(video_path, output_path, duration_sec=30):
    """
    Compute and visualize dense optical flow for first N seconds of video.
    
    Uses Farneback's algorithm (polynomial expansion) for dense flow estimation.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * duration_sec)
    
    print(f"Processing: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print(f"Analyzing first {duration_sec}s ({max_frames} frames)")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read video")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    flow_magnitudes = []
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute dense optical flow using Farneback
        # Parameters: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        # Visualize flow
        flow_vis = visualize_flow(flow)
        
        # Side-by-side: original + flow
        combined = np.hstack([frame, flow_vis])
        writer.write(combined)
        
        # Statistics
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(np.mean(mag))
        
        # Update
        prev_gray = gray
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{max_frames}, avg flow magnitude: {np.mean(mag):.2f}")
    
    cap.release()
    writer.release()
    
    print(f"\nOutput saved: {output_path}")
    print(f"Average flow magnitude across video: {np.mean(flow_magnitudes):.2f} pixels/frame")
    
    return flow_magnitudes


def visualize_flow(flow):
    """
    Convert flow to color image using HSV representation.
    
    Hue = direction (0-360 degrees)
    Saturation = magnitude (normalized)
    Value = 255 (constant brightness)
    """
    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # Hue: direction (0-180 in OpenCV)
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Saturation: normalized magnitude
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 1] = mag_norm.astype(np.uint8)
    
    # Value: full brightness
    hsv[..., 2] = 255
    
    # Convert to BGR for video
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr


def analyze_flow_characteristics(video_path, sample_frame=15):
    """
    Extract and explain flow characteristics at a specific frame.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Read to sample frame
    for _ in range(sample_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if _ == sample_frame - 1:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _ == sample_frame:
            curr_frame = frame
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cap.release()
    
    # Compute flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Analysis
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    print(f"\nFlow Analysis (Frame {sample_frame}):")
    print(f"  Magnitude range: [{mag.min():.2f}, {mag.max():.2f}] pixels")
    print(f"  Mean magnitude: {mag.mean():.2f} pixels")
    print(f"  Std magnitude: {mag.std():.2f} pixels")
    
    # Direction analysis
    print(f"\n  Direction distribution:")
    print(f"    Right (0°): {np.sum((ang >= 0) & (ang < np.pi/4)) / ang.size * 100:.1f}%")
    print(f"    Up (90°): {np.sum((ang >= np.pi/4) & (ang < 3*np.pi/4)) / ang.size * 100:.1f}%")
    print(f"    Left (180°): {np.sum((ang >= 3*np.pi/4) & (ang < 5*np.pi/4)) / ang.size * 100:.1f}%")
    print(f"    Down (270°): {np.sum((ang >= 5*np.pi/4) & (ang < 7*np.pi/4)) / ang.size * 100:.1f}%")
    
    # Save analysis image
    flow_vis = visualize_flow(flow)
    overlay = cv2.addWeighted(curr_frame, 0.6, flow_vis, 0.4, 0)
    cv2.imwrite(f"flow_analysis_frame{sample_frame}.png", overlay)
    print(f"\nSaved: flow_analysis_frame{sample_frame}.png")
    
    return flow, mag, ang


if __name__ == "__main__":
    # Process two videos
    videos = [
        ("data/video1.mov", "output_flow_video1.mp4"),
        ("data/video2.mov", "output_flow_video2.mp4")
    ]
    
    for input_path, output_path in videos:
        if os.path.exists(input_path):
            print("="*60)
            magnitudes = compute_optical_flow_video(input_path, output_path, duration_sec=30)
            
            # Detailed analysis on middle frame
            analyze_flow_characteristics(input_path, sample_frame=15)
        else:
            print(f"Video not found: {input_path}")
