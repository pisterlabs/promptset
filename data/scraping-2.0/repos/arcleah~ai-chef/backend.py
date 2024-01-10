from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import sqlite3
from openai import OpenAI
import json

app = Flask(__name__)
app.secret_key = 'corn'  # for session management
correct_email = 'johndoe123@gmail.com'
correct_password = '1234'
client = OpenAI(api_key="api key")

database = r"C:\Users\yo-s-\Documents\GitHub\ai-chef\sqlite\database.db"  # Define the database path here

@app.route("/", methods = ['GET', "POST"])
def test():
    if request.method == 'POST':
        return render_template("mainpage.html") # Connect to main page
    return render_template("mainpage.html")

@app.route("/aboutus") # Connect to about us page
def aboutus():
    return render_template("aboutus.html")

@app.route("/home") # Connect to home page
def home():
    return render_template("home.html")

@app.route("/login", methods=['GET', 'POST']) # Connect to login page
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == correct_email and password == correct_password: # Check if username/password entered matches correct email & password
            # Correct credentials go to pantry page
            return redirect(url_for('pantry'))
        else:
            # Incorrect credentials, display error
            flash('Invalid username or password', 'error')
            return render_template("login.html",)

    return render_template("login.html") # Go back to login page

@app.route("/signup") # Connect to signup page
def signup():
    return render_template("signup.html")

@app.route("/pantry") # Connect to pantry page
def pantry():
    return render_template("pantry.html")

@app.route('/generate-recipe', methods=['POST']) # Connect to generate-recipe
def generate_recipe():
    data = request.json
    items = data['itemsData']
    
    # Construct the prompt with items details
    prompt = "You are a professional chef, skilled in creating recipes. I only have limited items in the pantry, some that might be expiring soon such as:\n"
    for item in items:
        prompt += f"Produce: {item['produce']}, Quantity: {item['quantity']}, Expiry Date: {item['expiryDate']}\n"
    prompt += "Can you give me a detailed recipe I can make with these ingredients? Please ensure that you are focusing on the expiring ingredients from my pantry \n"
    prompt += "(i.e. 1 day until expiration) to lower food waste before creating the recipe. Please also make sure to list the times needed for each step, \n" 
    prompt += "the measurements (as an example, sear the steak for 2 mins on each side, boil the potatoes for 10 mins, etc), \n" 
    prompt += "the estimated time it will take to cook the meal (i.e. overall time is 30 mins, 1 hour, etc). \n"
    prompt += "please format it with the name of the recipe attached with the estimated time to prepare, followed by the list of ingredients and the amount needed, followed by the steps and make them detailed by including the amount of ingredients and time needed for each step"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    # Extract the recipe content from the response
    recipe_content = response.choices[0].message.content

    # Return the recipe content in JSON format
    return jsonify({'recipe': recipe_content})


if __name__ == "__main__":
    app.run()