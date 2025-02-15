from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
socketio = SocketIO(app)

# Model configuration
DEFAULT_IMAGE_SIZE = (128, 96)  # Fallback dimensions if model has dynamic input shape
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'current_model.keras')
model = None

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
    
    try:
        global model
        file.save(MODEL_PATH)
        model = load_model(MODEL_PATH, compile=False)
        
        # Get model summary
        import io
        import sys
        summary_io = io.StringIO()
        sys.stdout = summary_io
        model.summary()
        sys.stdout = sys.__stdout__
        
        return jsonify({
            'success': 'Model uploaded successfully',
            'model_summary': summary_io.getvalue()
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
            # Use model's input shape if valid, otherwise fall back to default size
            model_input_shape = model.input_shape[1:3]
            using_fallback = None in model_input_shape
            target_size = DEFAULT_IMAGE_SIZE if using_fallback else model_input_shape[::-1]
            
            # Show what input shape is being used
            message = (
                f"Model input shape: {model.input_shape}\n"
                f"Using {'fallback' if using_fallback else 'model'} dimensions: {target_size}"
            )
            emit('model_info', {'message': message})
            
            # Resize frame
            processed_frame = cv2.resize(frame, target_size)
            
            processed_frame = np.expand_dims(processed_frame, axis=0)
            prediction = model.predict(processed_frame, verbose=0)
            
            # Get predictions for all classes
            class_names = ['paper', 'rock', 'scissors']
            probabilities = [float(p) for p in prediction[0]]
            class_idx = np.argmax(probabilities)
            
            emit('prediction', {
                'classes': class_names,
                'probabilities': probabilities,
                'highest_class': class_names[class_idx]
            })
        except Exception as e:
            emit('error', {'message': str(e)})
    except Exception as e:
        emit('error', {'message': f'Frame processing error: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
