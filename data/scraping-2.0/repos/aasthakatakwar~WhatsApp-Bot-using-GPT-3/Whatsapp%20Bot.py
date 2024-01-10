from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import openai
import os

app = Flask(__name__)
openai.api_key = "Key"

# Initialize conversation history as empty list
memory = []

def get_openai_response(incoming_msg):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=(f"{memory[-1]}\n" if len(memory) > 0 else "") +
               f"User: {incoming_msg}\nBot:",
        temperature=0.1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()
@app.route('/cb', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    if not memory:
        memory.append(incoming_msg)
        prompt = f"I am a chatbot. You said: {incoming_msg}. What can I help you with today?"
    else:
        memory.append(incoming_msg)
        prompt = f"Conversation so far: {' >> '.join(memory)}. What can I help you with now?"

    msg.body(get_openai_response(incoming_msg))
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)
