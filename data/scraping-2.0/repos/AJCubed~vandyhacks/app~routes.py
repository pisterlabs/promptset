# Import necessary libraries
from app import app, db
from flask import Flask, render_template, request, redirect
from sqlalchemy import create_engine, false, true
import json
import openai
import os
import time
from dotenv import load_dotenv
from app.models import User_Hist

db.create_all()
engine = create_engine('sqlite:///app/userdata.sqlite3', echo=False)

load_dotenv()

# Set the OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Define the name of the bot
name = 'BOT'

# Define the role of the bot
role = 'Hospital Front Desk'

# Set the OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Define the name of the bot
name = 'Amy'

# Define the role of the bot
role = 'Hospital Front Desk'

# Define the impersonated role with instructions
impersonated_role = f"""
    You are a hospital front desk AI agent. Patients are coming to you for help scheduling appointments for various illnesses. Be aware that you are dealing with personal and sensitive information.
    You must establish all of these details during the conversation. However, please speak at a natural pace, but professionally without reciting these questions word-for-word.
        1. "May I have your name and birthday to access your record?"
        2. What symptoms have they been having? Are there any others? Ask until they have no more issues to report.
        3. How long have symptoms been going on?  How severe are they (mild, moderate, severe)? 
        4. If they have any known allergies to medications, food, or other substances.
        5. Are there any chronic illnesses that run in their family? (e.g., diabetes, hypertension, heart diseases)
        6. Are they currently on any medications? If so, what are they for?
        7. Have they been diagnosed with any related medical conditions in the recent past?
        8. Suggest a hospital department suitable for treating them (cardiology for breathing issues, etc), and ask if they would like to schedule an appointment.
        9. If they would like an appointment, ask for a preferred date and tell the user their appointment will be scheduled shortly. If not, cordially end the conversation.
"""

output_format = f"""
    Extract these details into a JSON file with this exact format:
        {{
            "name": string
            "birthday": date
            "symptoms" : "symptom1, symptom2(if existent)"
            "allergies" : string
            "chronic_illnesses" : string
            "doctor_status" : "no, or doctors"
            "medications" : "no, or medications"
            "past_diagnosis": string
            "summary": string
            "apptDate": date
        }}
    Birthday and apptDate should be of mm-dd-yy types (if it's entered as a word, convert it). The current year is 2023. 
    Symptoms, allergies, chronic_illnesses, medications,doctor_status, past_diagnosis,  should be only keywords, and summary should be a short paragraph of the entire conversation, symptoms, important information, severity, and duration for suggested doctor reference.
    Do not write anything other than this JSON file.
"""

# Initialize variables for chat history
explicit_input = ""
chatgpt_output = 'Chat log: /n'
cwd = os.getcwd()
chat_histories_folder = os.path.join(cwd, 'chat_histories')

# Ensure the chat_histories folder exists or create it
if not os.path.exists(chat_histories_folder):
    os.makedirs(chat_histories_folder)

i = 1

# Find an available chat history file
while os.path.exists(os.path.join(chat_histories_folder, f'chat_history{i}.txt')):
    i += 1

history_file = os.path.join(chat_histories_folder, f'chat_history{i}.txt')

# Create a new chat history file
with open(history_file, 'w') as f:
    f.write('\n')

# Initialize chat history
chat_history = ''

# Function to complete chat input using OpenAI's GPT-3.5 Turbo
def chatcompletion(user_input, impersonated_role, explicit_input, chat_history):
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": f"{impersonated_role}. Conversation history: {chat_history}"},
            {"role": "user", "content": f"{user_input}. {explicit_input}"},
        ]
    )

    for item in output['choices']:
        chatgpt_output = item['message']['content']

    return chatgpt_output

# Function to handle user chat input
def chat(user_input):
    global chat_history, name, chatgpt_output
    current_day = time.strftime("%d/%m", time.localtime())
    current_time = time.strftime("%H:%M:%S", time.localtime())
    chat_history += f'\nUser: {user_input}\n'
    chatgpt_raw_output = chatcompletion(user_input, impersonated_role, explicit_input, chat_history).replace(f'{name}:', '')
    chatgpt_output = f'{chatgpt_raw_output}'
    chat_history += chatgpt_output + '\n'
    with open(history_file, 'a') as f:
        f.write('\n'+ current_day+ ' '+ current_time+ ' User: ' +user_input +' \n' + current_day+ ' ' + current_time+  ' ' +  chatgpt_output + '\n')
        f.close()
    return chatgpt_raw_output

# Function to get a response from the chatbot
def get_response(userText):
    if "thank you" in userText.lower():
        generate_json_summary(history_file)
        return "No problem!"
    return chat(userText)

def generate_json_summary(filename):
    with open(filename, 'r') as f:
        content = f.read()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Read this following conversation transcript\n\n{content}\n\nOutput format: {output_format}"}
    ]

    output = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=2000,
        messages=messages
    )

    output = output.choices[0].message["content"]
    start = output.find("{")
    end = output.rfind("}")+1 
    output = output[start:end]

    
    GPT_res = json.loads(output)
    print(GPT_res)

    user_name = GPT_res["name"]
    user_birthday = GPT_res["birthday"]
    user_symptoms = GPT_res["symptoms"]
    user_chronic_illnesses = GPT_res["chronic_illnesses"]
    user_doctor_status = GPT_res["doctor_status"]
    user_allergies = GPT_res["allergies"]
    user_medications = GPT_res["medications"]
    user_summary = GPT_res["summary"]
    user_past_diagnosis = GPT_res["past_diagnosis"]
    user_apptDate = GPT_res["apptDate"]

    User_entry = User_Hist(user_name, user_birthday, user_symptoms, user_allergies, user_chronic_illnesses, 
                           user_doctor_status, user_medications, user_summary, user_past_diagnosis, user_apptDate)

    db.session.add(User_entry)
    db.session.commit()


# Define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/database")
def database():
    qresults = User_Hist.query.all()
    
    return render_template("database.html", data = qresults)

@app.route("/get")
# Function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))

@app.route('/refresh')
def refresh():
    time.sleep(600) # Wait for 10 minutes
    return redirect('/refresh')

# Run the Flask app
if __name__ == "__main__":
    generate_json_summary("/chat_histories/chat_history1.txt")
    # app.run()
