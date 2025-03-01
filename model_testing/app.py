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
        
        # Get model summary and output shape info
        import io
        import sys
        summary_io = io.StringIO()
        
        # Get basic model summary
        sys.stdout = summary_io
        model.summary()
        sys.stdout = sys.__stdout__
        
        # Add output shape information and detect encoding type
        output_shape = model.output_shape[-1]
        global is_categorical
        is_categorical = output_shape > 1
        summary_io.write(f"\nModel Output Classes: {output_shape}")
        summary_io.write(f"\nEncoding Type: {'Categorical' if is_categorical else 'Integer'}\n")
        
        # Try to get class indices if available
        try:
            if hasattr(model, 'get_config'):
                config = model.get_config()
                if isinstance(config, dict) and 'class_indices' in config:
                    class_indices = config['class_indices']
                    summary_io.write(f"Class Names: {', '.join(sorted(class_indices.keys()))}\n")
        except Exception as e:
            summary_io.write("Note: Class names not stored in model configuration\n")
            
        # Store the number of output classes for predictions
        global class_names
        class_names = ['paper', 'rock', 'scissors']  # Default fallback
        if output_shape == len(class_names):
            summary_io.write(f"\nUsing default class mapping: {', '.join(class_names)}\n")
        else:
            summary_io.write(f"\nWarning: Model output shape ({output_shape}) doesn't match default classes ({len(class_names)})\n")
        
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
            
            # Use global class names for predictions
            global class_names, is_categorical
            
            if is_categorical:
            probabilities = [float(p) for p in prediction[0]]
            class_idx = np.argmax(probabilities)
                confidence = probabilities[class_idx]
            else:
                class_idx = int(prediction[0][0])
                probabilities = [1.0 if i == class_idx else 0.0 for i in range(len(class_names))]
                confidence = 1.0
            
            emit('prediction', {
                'classes': class_names,
                'probabilities': probabilities,
                'highest_class': class_names[class_idx],
                'encoding_type': 'categorical' if is_categorical else 'integer',
                'confidence': confidence
            })
        except Exception as e:
            emit('error', {'message': str(e)})
    except Exception as e:
        emit('error', {'message': f'Frame processing error: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
