# Lane-line Detection

This repository contains a Python script for detecting lane lines in a video using OpenCV and NumPy. The script processes each frame of the video to identify lane markings and overlay them on the original video.

## Features

- **Lane Detection**: Detects lane lines in a video using edge detection and Hough line transform.
- **Region of Interest**: Applies a mask to focus on the relevant portion of the frame.
- **Average Slope Intercept**: Computes average lines for better lane line representation.
- **Command Line Interface**: Easily pass the video file path through CLI for processing.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Optimus-Q/Lane-line.git
    cd lane-detection
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the lane detection script, use the following command:

```sh
python lane_detection.py path/to/your/video.mp4
