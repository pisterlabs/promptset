from flask import Flask, render_template, request
from openai import OpenAI
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

app = Flask(__name__)


# Replace the following with your Google Drive link
google_drive_link = "https://docs.google.com/spreadsheets/d/1MPUIMXgPqsc81olEYST_ehBTSnNvptbS/export?format=xlsx"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

df = pd.read_excel(google_drive_link)
df_json = df.to_json(orient='split')
system_message = {"role": "system", "content": df_json}

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/chat', methods=['POST'])
def chat_bot():
    user_input = request.form['user_input']

    # Check if the user wants to quit
    if user_input.lower() == 'quit':
        return render_template('index3.html', user_input=user_input, assistant_response="Chatbot session ended.")

    # Add user input to the conversation
    user_messages = [{"role": "user", "content": user_input}]
    conversation = [system_message] + user_messages

    # Get the completion from OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Display assistant's response
    assistant_response = completion.choices[0].message.content

    # Update the conversation with the assistant's response
    user_messages.append({"role": "assistant", "content": assistant_response})
    conversation = [system_message] + user_messages

    # Store the conversation in the database
    model_info = "gpt-3.5-turbo"  # Modify this based on the actual model used
    timestamp = datetime.utcnow()
    app.logger.info(f"Conversation: {conversation}")
    app.logger.info(f"Model Info: {model_info}")
    app.logger.info(f"Timestamp: {timestamp}")

    return render_template('index3.html', user_input=user_input, assistant_response=assistant_response)

if __name__ == '__main__':
    app.run(debug=True)
