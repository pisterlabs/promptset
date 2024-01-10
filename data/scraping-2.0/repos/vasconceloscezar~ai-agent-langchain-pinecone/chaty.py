import os
import openai
import socketio
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Set up OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up Socket.IO client
sio = socketio.Client()

message_history = []
initial_prompt = "You are Andrew, a Customer Support agent. Your task is to guide and help the user through the system."


def generate_response(message):
    new_message = f"\nUser Message: {message}  \nCustomer Support(Andrew):"
    prompt = initial_prompt + "".join(message_history) + new_message
    current_time_and_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["(Andrew):"],
        )

        response = (
            completions.choices[0].text
            if completions.choices[0].text
            else "Thanks for reaching out to us, we will get back to you soon."
        )

        message_history.append(prompt)
        message_history.append(response)
        print(f"({current_time_and_date}) PROMPT: {prompt}\nRESPONSE: {response}")
        return response
    except Exception as error:
        print(f"({current_time_and_date}) PROMPT: {prompt} ERROR: {error}")
        return "Sorry, I am having trouble generating a response right now. Please try again later."


@sio.event
def connect_error(error):
    print(f"Connection error: {error}")


@sio.event
def message(data):
    if data["user"] == "Bot":
        return
    print(f'({data["user"]}) said:', data["message"])
    response = generate_response(data["message"])
    sio.emit("message", {"user": "Bot", "message": response})


def send_message(message):
    if message.strip():
        sio.emit("message", {"user": "Bot", "message": message})


if __name__ == "__main__":
    sio.connect("http://localhost:8080")

    try:
        while True:
            message = input("Enter message: ")
            send_message(message)
    except KeyboardInterrupt:
        sio.disconnect()
