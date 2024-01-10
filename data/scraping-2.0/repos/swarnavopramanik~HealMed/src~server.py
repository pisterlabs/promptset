from flask import Flask, request, jsonify
import json
import openai

app = Flask(__name__)

# Load doctor details from JSON file
with open('C:/Users/godli/OneDrive/Desktop/HELPMED/healmed/src/Components/Database/doctor_details.json') as f:
    doctor_details = json.load(f)

# OpenAI API configuration
openai.api_key = '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'

# Define the function to interact with the bot
def chat_with_doctor_bot(message):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"User: {message}\nBot:",
        temperature=0.6,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

# API route for chat
@app.route('/api/chat', methods=['POST','GET'])
def chat():
    if request.headers['Content-Type'] == 'application/json':
        message = request.json['message']
        bot_response = chat_with_doctor_bot(message)

        if 'Dr.' in bot_response:
            doctor_name = bot_response.split('Dr.')[1].split('.')[0].strip()
            doctor = next((d for d in doctor_details if d['name'].lower() == doctor_name.lower()), None)
            if doctor:
                response_data = {
                    'botResponse': f"Bot: Here is the information about {doctor['name']}:\n"
                    f"ID: {doctor['id']}\n"
                    f"Specialization: {doctor['specialization']}\n"
                    f"Contact Number: {doctor['contact_number']}\n"
                    f"Email: {doctor['email']}"
                }
            else:
                response_data = {'botResponse': "Bot: Sorry, I don't have information about that doctor."}
        else:
            response_data = {'botResponse': f"Bot: {bot_response}"}
        print(respond_data)
    else: 
        return jsonify({'error': 'Unsupported Media Type'}), 415

if __name__ == '__main__':
    app.run()