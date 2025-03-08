from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import os
import json

app = Flask(__name__)
socketio = SocketIO(app)

# Model configuration - updated to match hp_tuner_standalone.py
IMAGE_DIMS = (240, 320)  # Match hp_tuner IMAGE_DIMS 
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'current_model.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')
model = None
class_names = ['rock', 'paper', 'scissors']  # Default order from hypertuner

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model']
    if file.filename == '' or not file.filename.endswith('.keras'):
        return jsonify({'error': 'Invalid file format. Please upload .keras file'}), 400
    
    # Handle optional class names file
    if 'class_names' in request.files:
        class_file = request.files['class_names']
        if class_file.filename != '':
            try:
                class_file.save(CLASS_NAMES_PATH)
                with open(CLASS_NAMES_PATH, 'r') as f:
                    global class_names
                    class_names = json.load(f)
            except Exception as e:
                return jsonify({'error': f'Failed to load class names: {str(e)}'}), 400
    
    try:
        global model
        file.save(MODEL_PATH)
        model = load_model(MODEL_PATH, compile=False)
        
        # Get model summary and output shape info
        import io
        import sys
        summary_io = io.StringIO()
        
        # Get basic model summary
        sys.stdout = summary_io
        model.summary()
        sys.stdout = sys.__stdout__
        
        # Get input and output shape information
        input_shape = model.input_shape[1:]
        output_shape = model.output_shape[-1]
        
        summary_io.write(f"\nModel Input Shape: {input_shape}")
        summary_io.write(f"\nModel Output Classes: {output_shape}")
        summary_io.write(f"\nUsing class names: {', '.join(class_names)}")
        
        return jsonify({
            'success': 'Model uploaded successfully',
            'model_summary': summary_io.getvalue(),
            'class_names': class_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('frame')
def handle_frame(frame_data):
    if model is None:
        emit('error', {'message': 'No model loaded'})
        return
        
    try:
        # Decode frame
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess and predict
        try:
            # Resize to match the model's expected input dimensions from hp_tuner
            # Note: OpenCV uses (width, height) order, but our constants are (height, width)
            processed_frame = cv2.resize(frame, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            
            # Normalize pixel values as done in hp_tuner
            processed_frame = processed_frame.astype(np.float32) / 255.0
            
            # Add batch dimension
            processed_frame = np.expand_dims(processed_frame, axis=0)
            
            # Make prediction
            prediction = model.predict(processed_frame, verbose=0)
            
            # Process prediction (always categorical for hp_tuner models)
            probabilities = [float(p) for p in prediction[0]]
            class_idx = np.argmax(probabilities)
            confidence = probabilities[class_idx]
            
            # Use global class names for predictions
            global class_names
            
            emit('prediction', {
                'classes': class_names,
                'probabilities': probabilities,
                'highest_class': class_names[class_idx],
                'encoding_type': 'categorical',
                'confidence': confidence
            })
        except Exception as e:
            emit('error', {'message': f'Prediction error: {str(e)}'})
    except Exception as e:
        emit('error', {'message': f'Frame processing error: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)