"""
module2/get_coordinates.py
Helper script to get pixel coordinates of A4 paper corners and object points
in an image using mouse clicks.
"""

import cv2 # OpenCV for computer vision tasks


# Load the image where user will click points

# img = cv2.imread("data/object_to_measure.jpg")
img = cv2.imread("data/validation_image.jpg")
if img is None:
    print("Error: Image not found")
    exit()


coords = [] # To store clicked coordinates


def mouse_click(event, x, y, flags, param):
    """
    Mouse callback to record clicks
    
    :param event: The type of mouse event
    :param x: X-coordinate of the mouse event
    :param y: Y-coordinate of the mouse event
    :param flags: Any relevant flags passed by OpenCV
    :param param: Additional parameters
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append([x, y])
        print(f"Point {len(coords)}: [{x}, {y}]")
        
        # Draw on image
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(len(coords)), (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Click 4 A4 corners + 2 object points", img)


cv2.imshow("Click 4 A4 corners + 2 object points", img)
cv2.setMouseCallback("Click 4 A4 corners + 2 object points", mouse_click)


print("Click in this order:")
print("1. A4 Top-Left, 2. A4 Top-Right, 3. A4 Bottom-Right, 4. A4 Bottom-Left")
print("5. Object Point 1, 6. Object Point 2")
print("Then press any key...")


cv2.waitKey(0)
cv2.destroyAllWindows()


print("\nCopy these into measure_object.py:")
print(f"ref_image_points = {coords[:4]}")
print(f"object_pt1 = {coords[4]}")
print(f"object_pt2 = {coords[5]}")
