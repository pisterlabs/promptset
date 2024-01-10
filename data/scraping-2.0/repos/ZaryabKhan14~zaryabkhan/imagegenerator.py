from flask import Flask, request, render_template, Blueprint, session, redirect, url_for, make_response
import os
from openai import OpenAI
from dotenv import load_dotenv

# Create a Flask Blueprint
imagegenerator_app = Blueprint('imagegenerator', __name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("api_key")

# Set a secret key for session management
imagegenerator_app.secret_key = os.getenv('zaryabkhan')  # Use an environment variable for the secret key

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Function to generate and display the image
def generate_and_display_image(text, model="dall-e-3", size="1024x1024", quality="standard"):
    try:
        response = client.images.generate(
            model=model,
            prompt=text,
            size=size,
            quality=quality,
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Route for the main page
@imagegenerator_app.route('/imagegenerator', methods=['GET', 'POST'])
def index():
    # Redirect to login if not logged in
    if 'username' not in session:
        return redirect(url_for('imagegenerator.login'))

    if request.method == 'POST':
        user_input = request.form.get('prompt')
        image_url = generate_and_display_image(user_input)
        response = make_response(render_template('imagegenerator.html', image_url=image_url))
    else:
        response = make_response(render_template('imagegenerator.html', image_url=None))
    
    # Set headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Login route
@imagegenerator_app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        # Redirect to main page if already logged in
        return redirect(url_for('imagegenerator.index'))

    if request.method == 'POST':
        # Hardcoded credentials for demonstration, consider using a database for production
        user_name = "zaryab"
        user_password = "zaryab"
        if user_name == request.form.get('username') and user_password == request.form.get('password'):
            session['username'] = user_name
            return redirect(url_for('imagegenerator.index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    else:
        return render_template('login.html')

# Logout route
@imagegenerator_app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('imagegenerator.login'))
