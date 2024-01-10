from flask import Flask, render_template, request
import openai
from dotenv import load_dotenv
import os  # Add this line
# Load environment variables
load_dotenv()

app = Flask(__name__)  # Use __name__, not _name_

# Set up OpenAI API credentials
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the default route to return the index.html file
@app.route("/")
def index():
    return render_template("index.html")

# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    # Get the message from the POST request
    user_message = request.json.get("message")
    
    # Define a system message to set context as medical
    system_message = {
    "role": "system",
    "content": "You are strictly a medical chatbot. Do not provide information outside of the medical domain. If a question isn't medical, inform the user and ask for a medical question."
}

    
    # Send the system message and user message to OpenAI's API and receive the response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            system_message,
            {"role": "user", "content": user_message}
        ]
    )
    
    response = completion.choices[0].message
    return response

if __name__ == '__main__':  # Use __name__, not _name_
    app.run()
else:
    print("You can only ask about Medical Related Questions")
