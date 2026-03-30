"""
Classroom 3D Mapping with YOLO + Stereo
Detects tables/chairs with YOLO, localizes with stereo vision
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


class YOLOStereoMapper:
    def __init__(self, baseline=0.12, focal_length=800): # these values are not correct, I just used the default values
        """
        Initialize YOLO detector and stereo parameters.
        """
        # Load YOLOv8 (pre-trained on COCO, fine-tuned for furniture if needed)
        print("Loading YOLO model...")
        self.model = YOLO('yolov8s.pt')  # 'yolov8n.pt' is the default model, 'yolov8s.pt' for better accuracy
        
        # Stereo parameters
        self.baseline = baseline  # meters
        self.focal_length = focal_length  # pixels
        
        # For stereo matching within detected regions
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*10,
            blockSize=11,
            P1=8*3*11**2,
            P2=32*3*11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2
        )
        
    def detect_objects(self, img):
        """
        Run YOLO detection on image.
        Returns: list of (class_name, bbox, confidence)
        """
        results = self.model(img, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                # Filter for furniture classes
                if cls_name in ['dining table', 'chair', 'bench', 'sofa', 'book']:
                    detections.append((cls_name, bbox, conf))
        
        return detections
    
    def compute_object_disparity(self, img_left, img_right, bbox):
        """
        Compute disparity for object region using stereo matching.
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract regions
        left_roi = img_left[y1:y2, x1:x2]
        right_roi = img_right[y1:y2, x1:x2]
        
        if left_roi.size == 0 or right_roi.size == 0:
            return None
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Get median disparity (robust to noise)
        valid_disp = disparity[disparity > 0]
        if len(valid_disp) == 0:
            return None
        
        median_disp = np.median(valid_disp)
        
        return median_disp
    
    def triangulate(self, bbox, disparity, img_shape):
        """
        Convert detection to 3D coordinates.
        Uses center of bounding box.
        """
        if disparity is None or disparity <= 0:
            return None
        
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        img_cx, img_cy = img_shape[1] // 2, img_shape[0] // 2
        
        # Z = f * B / disparity
        Z = (self.focal_length * self.baseline) / disparity
        
        # X = (x - cx) * Z / f
        X = (cx - img_cx) * Z / self.focal_length
        
        # Y = (y - cy) * Z / f  
        Y = (cy - img_cy) * Z / self.focal_length
        
        return np.array([X, Y, Z])
    
    def process_stereo_pair(self, img_left, img_right):
        """
        Full pipeline: detect, localize, generate floor plan.
        """
        print("Detecting objects in left image...")
        detections = self.detect_objects(img_left)
        
        print(f"Found {len(detections)} furniture objects")
        
        # Classify and localize
        tables_3d = []
        chairs_3d = []
        
        for cls_name, bbox, conf in detections:
            print(f"  {cls_name}: conf={conf:.2f}, bbox=[{bbox.astype(int)}]")
            
            # Compute 3D position
            disparity = self.compute_object_disparity(img_left, img_right, bbox)
            point_3d = self.triangulate(bbox, disparity, img_left.shape)
            
            if point_3d is None:
                print(f"    Failed to triangulate")
                continue
            
            print(f"    3D position: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})m")
            
            # Classify
            if cls_name in ['dining table', 'bench']:
                tables_3d.append(point_3d)
            elif cls_name in ['chair', 'book']:
                chairs_3d.append(point_3d)
        
        # Convert to arrays
        tables = np.array(tables_3d) if tables_3d else np.zeros((0, 3))
        chairs = np.array(chairs_3d) if chairs_3d else np.zeros((0, 3))
        
        return tables, chairs, detections
    
    def visualize_detection(self, img, detections, tables, chairs, output_path='detection_result.jpg'):
        """Draw detections and 3D positions on image."""
        img_vis = img.copy()
        
        for cls_name, bbox, conf in detections:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if cls_name in ['dining table', 'bench'] else (255, 0, 0)
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img_vis, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, img_vis)
        print(f"Saved detection visualization: {output_path}")
        
        return img_vis
    
    def plot_floor_plan(self, tables, chairs, output_path='floor_plan.png'):
        """
        Generate 2D floor plan: X-Y projection.
        Tables=Red circles, Chairs=Blue circles.
        """
        plt.figure(figsize=(12, 10))
        
        # Plot tables in red
        if len(tables) > 0:
            plt.scatter(tables[:, 0], tables[:, 1],
                       c='red', s=500, marker='s', alpha=0.7,
                       edgecolors='black', linewidth=2,
                       label=f'Tables ({len(tables)})')
            
            # Add labels
            for i, (x, y, z) in enumerate(tables):
                plt.annotate(f'T{i+1}\n({x:.2f}, {y:.2f})m',
                           (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8)
        
        # Plot chairs in blue
        if len(chairs) > 0:
            plt.scatter(chairs[:, 0], chairs[:, 1],
                       c='blue', s=300, marker='o', alpha=0.7,
                       edgecolors='black', linewidth=2,
                       label=f'Chairs ({len(chairs)})')
            
            for i, (x, y, z) in enumerate(chairs):
                plt.annotate(f'C{i+1}',
                           (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=8)
        
        plt.xlabel('X (meters)', fontsize=12)
        plt.ylabel('Y (meters)', fontsize=12)
        plt.title('Classroom Floor Plan: Tables (Red) and Chairs (Blue)', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Add origin marker
        plt.scatter([0], [0], c='green', s=200, marker='*', 
                   edgecolors='black', linewidth=2, label='Camera', zorder=5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved floor plan: {output_path}")
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("FLOOR PLAN SUMMARY")
        print("="*50)
        print(f"Camera position: (0, 0, ~1.2m)")
        print(f"Tables detected: {len(tables)}")
        for i, (x, y, z) in enumerate(tables):
            print(f"  Table {i+1}: ({x:.2f}, {y:.2f})m, distance={np.sqrt(x**2+y**2):.2f}m")
        print(f"Chairs detected: {len(chairs)}")
        for i, (x, y, z) in enumerate(chairs):
            print(f"  Chair {i+1}: ({x:.2f}, {y:.2f})m, distance={np.sqrt(x**2+y**2):.2f}m")
        print("="*50)


def main():
    # Configuration
    STEREO_BASELINE = 0.15  # ~15cm, adjusted to setup
    FOCAL_LENGTH = 4237.1      # pixels, adjusted to camera via calibration specs
    
    # Load stereo images
    img_left = cv2.imread('data/classroom_left.jpg')
    img_right = cv2.imread('data/classroom_right.jpg')
    
    if img_left is None or img_right is None:
        print("Error: Could not load images")
        print("Expected: data/classroom_left.jpg, data/classroom_right.jpg")
        return
    
    print(f"Loaded stereo pair: {img_left.shape}")
    
    # Process
    mapper = YOLOStereoMapper(baseline=STEREO_BASELINE, focal_length=FOCAL_LENGTH)
    
    tables, chairs, detections = mapper.process_stereo_pair(img_left, img_right)
    
    # Visualizations
    mapper.visualize_detection(img_left, detections, tables, chairs)
    mapper.plot_floor_plan(tables, chairs)
    
    # Save numerical results
    np.savez('classroom_3d_positions.npz', tables=tables, chairs=chairs)
    print("Saved: classroom_3d_positions.npz")


if __name__ == "__main__":
    main()
