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
            processed_frame = cv2.resize(frame, model.input_shape[1:3][::-1])
            processed_frame = np.expand_dims(processed_frame, axis=0)
            prediction = model.predict(processed_frame, verbose=0)
            
            # Get class and confidence
            class_names = ['paper', 'rock', 'scissors']
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            
            emit('prediction', {
                'class': class_names[class_idx],
                'confidence': confidence
            })
        except Exception as e:
            emit('error', {'message': str(e)})
    except Exception as e:
        emit('error', {'message': f'Frame processing error: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
