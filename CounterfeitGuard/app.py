"""
Flask API for Fake Currency Detection with Grad-CAM Visualization
"""
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import os
import io
import matplotlib.pyplot as plt
from matplotlib import cm
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///currency_detector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to access this page.'
login_manager.login_message_category = 'info'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store model
model = None
class_names = ['Fake', 'Genuine']


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_file():
    """Load the trained model"""
    global model
    
    # Only load if not already loaded
    if model is not None:
        return
    
    model_path = 'model/currency_detector.h5'
    
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        # Create a simple demo model if no trained model exists
        print("No trained model found. Creating demo model...")
        from CounterfeitGuard.model import create_model
        model = create_model()
        print("Demo model created. Train a real model for better accuracy.")


# Load model on app initialization
with app.app_context():
    load_model_file()


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array, img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap for model interpretability
    
    Args:
        img_array: Preprocessed image array
        model: Trained model
        last_conv_layer_name: Name of last convolutional layer (auto-detected if None)
        pred_index: Class index for which to compute Grad-CAM
    
    Returns:
        Heatmap array
    """
    # Find the MobileNetV2 base model layer
    base_model_layer = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model_layer = layer
            break
    
    # Get the last convolutional layer from the base model
    last_conv_layer = None
    if base_model_layer is not None:
        # Try to find the last convolutional layer in the base model
        try:
            last_conv_layer = base_model_layer.get_layer('out_relu')
        except:
            try:
                last_conv_layer = base_model_layer.get_layer('Conv_1')
            except:
                # Find any conv layer
                for layer in reversed(base_model_layer.layers):
                    if 'conv' in layer.name.lower() and hasattr(layer, 'output'):
                        last_conv_layer = layer
                        break
    
    # If we couldn't find a conv layer, use global average pooling as fallback
    if last_conv_layer is None:
        try:
            target_layer = model.get_layer('global_average_pooling2d')
        except:
            # Just use the layer before the final dense layer
            target_layer = model.layers[-3]
    else:
        # Create a new model that outputs both the conv layer and final predictions
        # We need to recreate the forward pass through the nested model
        target_layer = last_conv_layer
    
    # Build a model that returns the outputs of the target layer and the final predictions
    # Use a functional approach that works with nested models
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[base_model_layer.output if base_model_layer else model.layers[-3].output, model.output]
    )
    
    # Compute gradient of predicted class with respect to feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Gradient of output with respect to conv layer
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Avoid division by zero
    if grads is None:
        # Fallback: return a simple heatmap
        return np.ones((7, 7)) * 0.5
    
    # Mean intensity of gradient over specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        heatmap: Grad-CAM heatmap
        original_img: Original PIL image
        alpha: Transparency factor
        colormap: OpenCV colormap
    
    Returns:
        Overlaid image as bytes
    """
    # Convert PIL image to numpy array
    img_array = np.array(original_img)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    superimposed = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    
    # Convert to PIL Image and save to bytes
    result_img = Image.fromarray(superimposed)
    img_bytes = io.BytesIO()
    result_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken. Please choose another.', 'error')
            return render_template('register.html')
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'yes'
        
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    """Serve main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if currency is genuine or fake
    Returns JSON with prediction, confidence, and Grad-CAM visualization
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img_array, original_img = preprocess_image(filepath)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        
        # Generate Grad-CAM heatmap
        try:
            heatmap = make_gradcam_heatmap(img_array, model, pred_index=predicted_class)
            
            # Create overlay image
            gradcam_bytes = overlay_heatmap(heatmap, original_img)
            
            # Save Grad-CAM image
            gradcam_filename = f'gradcam_{filename}'
            gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            with open(gradcam_path, 'wb') as f:
                f.write(gradcam_bytes.read())
            
            gradcam_url = f'/gradcam/{gradcam_filename}'
        except Exception as gradcam_error:
            print(f"Grad-CAM error: {str(gradcam_error)}")
            import traceback
            traceback.print_exc()
            gradcam_url = None
        
        # Prepare response
        result = {
            'prediction': class_names[predicted_class],
            'confidence': round(confidence, 2),
            'is_genuine': bool(predicted_class == 1),
            'probabilities': {
                'fake': round(float(predictions[0][0]) * 100, 2),
                'genuine': round(float(predictions[0][1]) * 100, 2)
            }
        }
        
        if gradcam_url:
            result['gradcam_image'] = gradcam_url
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/gradcam/<filename>')
def get_gradcam(filename):
    """Serve Grad-CAM visualization image"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png')
    return jsonify({'error': 'File not found'}), 404


@app.route('/testing')
@login_required
def testing():
    """Serve testing page with model information"""
    return render_template('testing.html')


@app.route('/model-info')
@login_required
def model_info():
    """Return model information for testing"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get model summary
        layer_info = []
        for layer in model.layers:
            layer_info.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A'
            })
        
        # Check if model is trained or demo
        model_path = 'model/currency_detector.h5'
        is_trained = os.path.exists(model_path)
        
        return jsonify({
            'model_type': 'Trained Model' if is_trained else 'Demo Model (Random Weights)',
            'total_layers': len(model.layers),
            'layers': layer_info,
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'class_names': class_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database initialized successfully")
    
    # Load model at startup
    load_model_file()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
