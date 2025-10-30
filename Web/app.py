import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from PIL import Image

# --- APP & DATABASE CONFIGURATION ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'a_very_secret_key_that_you_must_change'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- DATABASE MODEL (User, History, etc. remain the same) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(10), default='user', nullable=False)
    
    date_of_birth = db.Column(db.String(20), nullable=True)
    location = db.Column(db.String(100), nullable=True)
    mobile_phone_no = db.Column(db.String(15), nullable=True)
    gender = db.Column(db.String(10), nullable=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

# --- DEEP LEARNING MODEL SETUP ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Model', 'model.h5')

# *** FIX APPLIED HERE ***
# MATCH THE IMAGE SIZE USED DURING TRAINING (150x150)
IMG_SIZE = 150 
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)

# Class names should match the training classes
# The training script saves these, but we hardcode them for robustness if the file is moved.
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def get_custom_objects():
    return {
        'TFOpLambda': tf.keras.layers.Lambda,
        'Lambda': tf.keras.layers.Lambda,
        'Conv2D': tf.keras.layers.Conv2D,
        'Dense': tf.keras.layers.Dense,
        'MaxPooling2D': tf.keras.layers.MaxPooling2D,
        'Flatten': tf.keras.layers.Flatten
    }

model = None
try:
    if os.path.exists(MODEL_PATH):
        # Load the model with compile=False to avoid issues if the optimizer is not available
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=get_custom_objects(), compile=False)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {MODEL_PATH}. Prediction will be disabled.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- AUTH HELPER ---
def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

# --- PREPROCESSING LAYER ---
def process_file(file):
    if model is None:
        raise RuntimeError('Deep learning model could not be loaded on the server.')

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # *** FIX APPLIED HERE: Resizing to match the model's expected input size ***
        img = img.resize(TARGET_SIZE)
        
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(np.max(probabilities)) * 100

        # Detailed probabilities for display
        all_probabilities = dict(zip(class_names, [f"{p*100:.2f}%" for p in probabilities]))

        return predicted_class_name, confidence, all_probabilities
        
    except Exception as e:
        # Re-raise error with context if needed for debugging, or return a generic message
        raise Exception(f'Exception encountered when calling Sequential.call(): {str(e)}')

# --- ROUTES ---

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Login failed. Please check your username and password.")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Retrieve all form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        date_of_birth = request.form.get('date_of_birth')
        location = request.form.get('location')
        mobile_phone_no = request.form.get('mobile_phone_no')
        gender = request.form.get('gender')
        
        if password != confirm_password:
            return "Passwords do not match."
            
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_password,
            date_of_birth=date_of_birth,
            location=location,
            mobile_phone_no=mobile_phone_no,
            gender=gender
        )
        try:
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for('login'))
        except:
            db.session.rollback()
            return "Registration failed. That username or email already exists."
    return render_template('registration.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', username=user.username)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('homepage'))

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password) and user.role == 'admin':
            session['user_id'] = user.id
            return redirect(url_for('admin_dashboard'))
        else:
            return "Admin login failed. Invalid credentials or insufficient permissions."
    return render_template('admin_login.html')

@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    user = User.query.get(session['user_id'])
    if user.role != 'admin':
        return "Access denied. You are not an administrator."
    all_users = User.query.all()
    return render_template('admin_dashboard.html', all_users=all_users)

@app.route('/predict-page')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/color-vision-test')
@login_required
def color_vision_test():  
    return render_template('colorvision_test.html')

@app.route('/eye-exercises-page')
@login_required
def eye_exercises_page():  
    return render_template('eye_exercises.html')

#@app.route('/profile-page')
#@login_required
#def profile_page():  
 #   return render_template('profile.html')
 
 # ... (rest of the code)

@app.route('/profile-page', methods=['GET', 'POST']) # <<< FIX: ADDED 'POST' METHOD
@login_required
def profile_page():
    user = User.query.get(session['user_id']) # Get the current user

    if request.method == 'POST':
        # --- Handle Profile Update (AJAX POST) ---
        try:
            # 1. Retrieve data from the form
            new_username = request.form.get('username')
            new_location = request.form.get('location')
            new_dob = request.form.get('date_of_birth')

            # 2. Basic Validation & Uniqueness Check (Crucial step)
            if not new_username:
                return jsonify({'success': False, 'message': 'Username is required.'}), 400

            # Check for username uniqueness (excluding the current user)
            username_exists = User.query.filter(
                User.username == new_username, 
                User.id != user.id
            ).first()
            
            if username_exists:
                return jsonify({'success': False, 'message': 'This username is already taken.'}), 400

            # 3. Update user object in the database
            user.username = new_username
            user.location = new_location
            user.date_of_birth = new_dob

            db.session.commit()

            # 4. Return success JSON response for the front-end AJAX
            return jsonify({
                'success': True, 
                'message': 'Profile updated successfully! ✅',
                'username': user.username # Optional: return new data
            })

        except Exception as e:
            db.session.rollback()
            # Log the error for debugging
            app.logger.error(f"Profile update failed for user {user.id}: {e}")
            return jsonify({
                'success': False, 
                'message': 'A database error occurred. Changes not saved.'
            }), 500

    # --- Handle Page Load (GET) ---
    # Convert date_of_birth to a string format expected by the HTML <input type="date">
    # Note: If date_of_birth is stored as a standard Python date object, 
    # you might need a different conversion (e.g., user.date_of_birth.isoformat() if it's a date object).
    # Since your model stores it as a String, this should work fine.
    return render_template('profile.html', user=user)

# New route to get user location
@app.route('/get_user_data')
@login_required
def get_user_data():
    user = User.query.get(session['user_id'])
    return jsonify({
        'success': True,
        'location': user.location if user.location else 'Unknown Location'
    })

# Placeholder for Hospital Search (uses Google Search API internally)
@app.route('/search_hospitals', methods=['POST'])
def search_hospitals():
    data = request.get_json()
    user_location = data.get('location', 'India')
    
    # Simulate a search (replace this with an actual API call later)
    query = f"top ophthalmology specialists or eye hospitals near {user_location}"
    
    # For now, return a placeholder result
    simulated_hospitals = [
        {"name": "VisionCare Center", "address": f"123 Main St, {user_location}", "rating": "4.8"},
        {"name": "Retina Specialist Group", "address": f"456 Oak Ave, {user_location}", "rating": "4.5"},
        {"name": "Local Eye Clinic", "address": f"789 Pine Ln, {user_location}", "rating": "4.2"}
    ]
    return jsonify({
        "success": True, 
        "query": query,
        "hospitals": simulated_hospitals
    })


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        return jsonify({'error': 'Prediction failed: The deep learning model could not be loaded on the server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        predicted_class_name, confidence, all_probabilities = process_file(file)
        
        # --- Actionable Step Logic (based on training class names) ---
        if predicted_class_name == "normal":
            action = "Healthy eyes detected. Great work!"
            reminder = "No immediate follow-up required. Continue regular check-ups."
        else:
            # Format the disease name for better display (e.g., 'diabetic_retinopathy' -> 'Diabetic Retinopathy')
            display_name = predicted_class_name.replace('_', ' ').title()
            action = f"**Urgent!** Specialist consultation recommended for potential {display_name}."
            reminder = f"Schedule a follow-up appointment with a specialist immediately."

        return jsonify({
            'success': True,
            'prediction': predicted_class_name,
            'display_name': display_name if predicted_class_name != 'normal' else 'Normal',
            'confidence': f"{confidence:.2f}%",
            'action': action,
            'reminder': reminder,
            'all_probabilities': all_probabilities
        })

    except Exception as e:
        # This will catch the incompatible shape error or any other processing error
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'Prediction: Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # --- TEMPORARY ADMIN CREATION CODE ---
        existing_admin = User.query.filter_by(role='admin').first()
        if not existing_admin:
            # Ensure the password here is the one you used for login
            hashed_password = bcrypt.generate_password_hash('your_admin_password').decode('utf-8')
            new_admin = User(username='admin', email='admin@example.com', password_hash=hashed_password, role='admin')
            db.session.add(new_admin)
            db.session.commit()
            print("Admin user created successfully!")
        else:
            print("Admin user already exists.")
        # --- END OF TEMPORARY CODE ---
        
    app.run(debug=True, port=5000)
