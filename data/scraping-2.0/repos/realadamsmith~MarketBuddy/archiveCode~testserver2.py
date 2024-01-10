from twilio.rest import Client
import openai
from flask import Flask, request, redirect

# Set up your Twilio account credentials
account_sid = ''
auth_token = ''
client = Client(account_sid, auth_token)

# Set up your OpenAI API key
openai.api_key = 'sk-'

# Set up a listener to receive incoming messages from Twilio
app = Flask(__name__)
print("hello World")

# Define a function to send a message to ChatGPT and return the response
@app.route('/sms', methods=['POST'])
def receive_message():
    # Get the incoming message from Twilio
    message = request.form['Body']
    print(message)
    phone_number = request.form['From']

    response = openai.Completion.create(
        engine="davinci",
        prompt=f"User: {message}\nAI:",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7
    )
    reply_message = response.choices[0].text.strip()
    
    # Create a Twilio MessagingResponse object to generate a reply message
    response = MessagingResponse()
    
    # Add the reply message to the Twilio MessagingResponse object
    response.message(reply_message)
    
    # Return the Twilio MessagingResponse object as a string
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
