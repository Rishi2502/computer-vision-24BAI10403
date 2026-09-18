import cv2
import numpy as np
import argparse


# ==============================
# CAMERA 1 PARAMETERS
# ==============================

K1 = np.array([
    [2216.47, 3.23465, 331.68],
    [0, 2219.88, 154.819],
    [0, 0, 1]
], dtype=np.float64)

R1 = np.array([
    [-0.537178, -0.116816, 0.835341],
    [0.0383902, -0.992723, -0.114137],
    [0.842595, -0.0292429, 0.537753]
], dtype=np.float64)

t1 = np.array([
    -1710.18,
    473.735,
    -1055.72
], dtype=np.float64)


# ==============================
# CAMERA 2 PARAMETERS
# ==============================

K2 = np.array([
    [2181.34, 9.07453, 330.694],
    [0, 2188.95, 192.278],
    [0, 0, 1]
], dtype=np.float64)

R2 = np.array([
    [-0.886458, -0.0106358, 0.462687],
    [-0.0552477, -0.990155, -0.128609],
    [0.4595, -0.139569, 0.877143]
], dtype=np.float64)

t2 = np.array([
    -925.754,
    483.655,
    -1860.54
], dtype=np.float64)


# ==============================
# CAMERA PROJECTION MATRICES
# ==============================

P1 = K1 @ np.hstack((
    R1.T,
    -R1.T @ t1.reshape(3, 1)
))

P2 = K2 @ np.hstack((
    R2.T,
    -R2.T @ t2.reshape(3, 1)
))


# ==============================
# COMMAND-LINE ARGUMENTS
# ==============================

parser = argparse.ArgumentParser(
    description="Face Detection and 3D Distance Estimation"
)

parser.add_argument(
    "--left",
    default="camera1_face.png",
    help="Path to left/camera 1 image"
)

parser.add_argument(
    "--right",
    default="camera2_face.png",
    help="Path to right/camera 2 image"
)

parser.add_argument(
    "--output",
    default="output.png",
    help="Path for the output image"
)

args = parser.parse_args()


# ==============================
# LOAD IMAGES
# ==============================

left = cv2.imread(args.left)
right = cv2.imread(args.right)

if left is None or right is None:
    print("Error: Could not load the stereo images.")
    print("Check the image paths.")
    exit()

print("Both stereo images loaded successfully!")


# ==============================
# FACE DETECTION
# ==============================

gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

faces_left = face_cascade.detectMultiScale(
    gray_left,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

faces_right = face_cascade.detectMultiScale(
    gray_right,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

print("Faces in camera 1:", len(faces_left))
print("Faces in camera 2:", len(faces_right))


if len(faces_left) == 0 or len(faces_right) == 0:
    print("Error: Face was not detected in both images.")
    exit()


# ==============================
# FACE CENTERS
# ==============================

x1, y1, w1, h1 = faces_left[0]
x2, y2, w2, h2 = faces_right[0]

center_left = (
    x1 + w1 / 2,
    y1 + h1 / 2
)

center_right = (
    x2 + w2 / 2,
    y2 + h2 / 2
)

print("Left face center:", center_left)
print("Right face center:", center_right)


# ==============================
# TRIANGULATION
# ==============================

points_left = np.array([
    [center_left[0]],
    [center_left[1]]
], dtype=np.float64)

points_right = np.array([
    [center_right[0]],
    [center_right[1]]
], dtype=np.float64)

point_4d = cv2.triangulatePoints(
    P1,
    P2,
    points_left,
    points_right
)

point_3d = point_4d[:3] / point_4d[3]

X = point_3d[0, 0]
Y = point_3d[1, 0]
Z = point_3d[2, 0]


# ==============================
# 3D DISTANCE
# ==============================

distance = np.sqrt(
    X**2 + Y**2 + Z**2
)

print("\n3D face position:")
print("X =", round(X, 2))
print("Y =", round(Y, 2))
print("Z =", round(Z, 2))

print(
    "\nEstimated 3D distance:",
    round(distance, 2),
    "dataset units"
)


# ==============================
# DRAW RESULT
# ==============================

cv2.rectangle(
    left,
    (x1, y1),
    (x1 + w1, y1 + h1),
    (0, 255, 0),
    2
)

text = f"Distance: {distance:.2f} units"

cv2.putText(
    left,
    text,
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)


# ==============================
# SAVE OUTPUT
# ==============================

cv2.imwrite(args.output, left)

print("\nOutput image saved to:", args.output)
print("Project completed successfully!")