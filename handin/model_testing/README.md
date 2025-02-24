# Model Testing Interface

A minimal web interface for testing deep learning models on webcam input.

## Requirements
- Flask
- Flask-SocketIO
- TensorFlow
- OpenCV (opencv-python-headless)
- NumPy

## Installation
First, install all required packages:
```bash
pip install -r requirements.txt
```

## Features
- Real-time webcam feed
- Upload and test .keras models
- Live predictions with confidence scores
- Error logging and model info display

## Usage
1. Run the application:
```bash
python app.py
```

2. Open http://localhost:5000 in your browser

3. Interface sections:
   - **Controls** (top-left): Upload model and control predictions
   - **Predictions** (top-right): Shows current prediction and confidence
   - **Webcam Feed** (bottom-left): Live camera input
   - **Logs** (bottom-right): Model information and error messages

## Notes
- Models must be in .keras format
- Default input shape for models is 224x224x3
- Ensure your webcam is accessible to the browser
