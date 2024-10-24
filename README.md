# OpenCV Wall-E

## Description
This project uses OpenCV to detect a predefined image of trash and estimate its distance from the camera. The system captures real-time video from the camera, detects the image using ORB feature matching, and calculates the distance to the object based on its size in the frame.

## Setup

1. Create a virtual environment and install the dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
