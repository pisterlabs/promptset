from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import openai
import os
from flask_cors import CORS
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with your secret key
socketio = SocketIO(app)
CORS(app)
openai.api_key = os.getenv("OPENAI_API_KEY")

connected_users = {}


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('connect_with_details')
def handle_connect_with_details(data):
    additional_data = data  # Access the additional details sent from the client
    user_info = {
        "email": "socket_id",
        'additional_data': additional_data,  # Store the additional details
        # Add any other user information you want to track
    }
    connected_users["socket_id"] = user_info
    print('Client connected. Socket ID:', "socket_id")


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


# signup/login login
userbase = {}

@socketio.on('gpt')
def handle_chat(data):
    input = data["input"]
    question = "so we are running a Restaurant Management app, we have a Restaurant website where the user comes login and then they redirect to the home page the inventory where all dishes are listed also on the same page we show user orders in the list where the user can see the order status in real-time like when order placed its status is 'Received' then it can update into 'Preparing', 'Delivered', 'Canceled' based on status. also user can buy online and we will deliver food to their doorstep. we serve our customers from morning 10 am to evening 10 pm, customer can book their table at our Restaurant by contacting us via phone call. delivery time is 1-3 hour. ``` So based on the above details you have about our business your task is to provide relevant answers to users, also make sure you feel them as you are human and also if you don't know the answer just keep giving round and round answer like a smart customer care representative. give them numbers so they can easily belive```"

    if len(data["chat"]):
        chat = data["chat"]
    else:
        chat = [
        {"role": "system", "content": question}
    ]

    chat.append({"role": "user", "content": input})

    print(chat)
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        max_tokens=60,
        n=1
    )
    chat.append(res.choices[0].message)
    text = res.choices[0].message
    emit("gpt", {"answer": text, "chat": chat})

if __name__ == '__main__':
    socketio.run(app, debug=True)
