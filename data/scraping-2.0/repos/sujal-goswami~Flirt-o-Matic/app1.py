from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv
import time

load_dotenv()  # Load variables from .env file into environment

app = Flask(__name__)

# Set your OpenAI GPT-3.5 API key as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace "OPENAI_API_KEY" with your actual API key

# Define the minimum time interval between API requests (in seconds)
MINIMUM_INTERVAL = 15  # Example: 15 seconds

last_request_time = None

def generate_pickup_line(user_input):
    global last_request_time
    
    if last_request_time is not None:
        elapsed_time = time.time() - last_request_time
        if elapsed_time < MINIMUM_INTERVAL:
            time.sleep(MINIMUM_INTERVAL - elapsed_time)
    
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=user_input,
        max_tokens=50,  # Adjust as needed
        temperature=0.7,
        n=1,
        stop=None,
    )
    
    last_request_time = time.time()
    pickup_line = response.choices[0].text.strip()
    return pickup_line


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        pickup_line = generate_pickup_line(user_input)
        return render_template("index.html", pickup_line=pickup_line)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)