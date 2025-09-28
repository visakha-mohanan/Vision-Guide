# Web/app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- APP & DATABASE CONFIGURATION ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'a_very_secret_key_that_you_must_change'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- DATABASE MODEL ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    role = db.Column(db.String(10), default='user', nullable=False)
    
    # --- ADDED NEW FIELDS ---
    date_of_birth = db.Column(db.String(20), nullable=True)
    location = db.Column(db.String(100), nullable=True)
    mobile_phone_no = db.Column(db.String(15), nullable=True)
    gender = db.Column(db.String(10), nullable=True)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

# --- DEEP LEARNING MODEL SETUP ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')
IMG_SIZE = 224
class_names = ["Normal", "Diabetic Retinopathy"]

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
    # Use a dummy model for now if the file doesn't exist to avoid crashing
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=get_custom_objects())
        print("Model loaded successfully.")
    else:
        print("Model file not found. Prediction will be disabled.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
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
def admin_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('admin_login'))
    user = User.query.get(session['user_id'])
    if user.role != 'admin':
        return "Access denied. You are not an administrator."
    all_users = User.query.all()
    return render_template('admin_dashboard.html', all_users=all_users)

@app.route('/predict-page')
def predict_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/color-vision-test')
def color_vision_test():   
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('colorvision_test.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Prediction failed: The deep learning model could not be loaded on the server. Please ensure the model file is valid and the environment is configured correctly.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    predictions = {}

    def process_file(file):
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            return predicted_class_name
        except Exception as e:
            return f'Error processing image: {str(e)}'

    predicted_class = process_file(file)
    predictions['prediction'] = predicted_class

    return jsonify(predictions)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # --- TEMPORARY ADMIN CREATION CODE ---
        # Only create a new admin if one doesn't exist
        existing_admin = User.query.filter_by(role='admin').first()
        if not existing_admin:
            hashed_password = bcrypt.generate_password_hash('your_admin_password').decode('utf-8')
            new_admin = User(username='admin', email='admin@example.com', password_hash=hashed_password, role='admin')
            db.session.add(new_admin)
            db.session.commit()
            print("Admin user created successfully!")
        else:
            print("Admin user already exists.")
        # --- END OF TEMPORARY CODE ---
        
    app.run(debug=True, port=5000)
