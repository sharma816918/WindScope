# WindScope

WindScope is a Python Flask project for wind-turbine video analysis
using classical computer-vision and digital image-processing methods.

## Features

- Video upload
- MP4, AVI, MOV, MKV and WEBM support
- Automatic moving-region detection
- Heuristic rotor-centre estimation
- Heuristic automatic blade-count estimation
- Motion-signal extraction
- Rotational-speed estimation
- Angular-frequency estimation
- Rotational-period estimation
- Blade-passage interval estimation
- Motion-signal graph
- ROI preview
- Video metadata and diagnostics
- Responsive orange-and-white interface

## Requirements

- Python 3.10 or later
- Flask
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Werkzeug

## Installation

Open the project directory in the terminal.

Install dependencies:

python -m pip install -r requirements.txt

## Run

python app.py

Open the browser:

http://127.0.0.1:5000

## Important Limitations

This project uses classical OpenCV methods rather than a trained AI model.

ROI detection and blade counting are heuristic.

The RPM estimate assumes that the dominant detected motion frequency
corresponds to blade-passage frequency.

Actual accuracy depends on video quality, camera stability, lighting,
rotor visibility and the validity of the detected blade count.

Results should be validated against independently measured RPM values.