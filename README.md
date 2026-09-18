# Face Detection and 3D Distance Estimation Using Stereo Vision

## 1. Project Overview

This project detects a human face in a pair of stereo images and estimates the 3D position and distance of the detected face using stereo vision and camera geometry.

The project combines:

- Face detection using OpenCV Haar Cascade
- Stereo camera calibration parameters
- Camera projection matrices
- Stereo triangulation
- 3D position estimation
- Euclidean distance calculation

## 2. How the Project Works

The system takes two images of the same subject captured from different camera viewpoints.

### Step 1: Face Detection

A Haar Cascade classifier detects the face in both camera images.

### Step 2: Find Face Centers

The center point of the detected face bounding box is calculated in each image.

### Step 3: Camera Projection

The intrinsic camera matrix, rotation matrix, and translation vector are used to construct the camera projection matrices.

### Step 4: 3D Triangulation

The corresponding face-center points from the two camera images are triangulated to obtain a 3D point.

### Step 5: Distance Estimation

The Euclidean distance of the 3D point from the camera coordinate origin is calculated.

The result is reported in dataset/calibration units because the physical unit of the supplied calibration translation values is not specified.

## 3. Technologies Used

- Python
- OpenCV
- NumPy

## 4. Project Structure

```text
CV PROJECT/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

## 5.Dataset and Input Images

This project uses the EPFL Stereo Face Database.

The dataset is not included in this GitHub repository because its distribution is restricted.

To run the project, obtain the EPFL Stereo Face Database and use one stereo image pair from the dataset.

For the default command, place the two input images in the project root directory with these exact filenames:

- `camera1_face.png`
- `camera2_face.png`

These should be the corresponding left and right images captured by Camera 1 and Camera 2.

The project can then be run using:

```bash
python main.py