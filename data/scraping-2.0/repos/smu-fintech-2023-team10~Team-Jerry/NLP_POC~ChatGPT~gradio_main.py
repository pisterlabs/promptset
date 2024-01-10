# Front-end test for Chat GPT chatbot to showcase the capabilities of ChatGPT for this use case #

import openai
import gradio as gr 
import os 

from constants import INTENT_DICT
from dotenv import load_dotenv
load_dotenv()

#OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")


#Generate Training String
training_str = ""
for intent, messages in INTENT_DICT.items():
    training_str += "Intent: " + intent + ", messages : "
    message_str = ",".join(messages) 
    training_str += message_str + ". End of Intent."

message_history = [
    {"role": "user", "content": """You are designed to be a prompt chatbot. I will give you a list of intents and messages that lead to those intent. 
If a user writes any prompts afterwards, only return the intents that I specify in the training intents and messages and if you do not know the intent, 
please return 'I do not know what you are speaking about, please try something along the lines of I want to transfer money or I want to check my account balance'. Training is included below.Training attached: """ + training_str},
    {"role": "assistant", "content": "Hello, I am your virtual chatbot assistant for OCBC Virtual banking. How may I help you today?"}
]


# Helper function for generating response
def chatbot_response(user_input):
    global message_history
    message_history.append({"role": "user", "content": user_input})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )

    reply_content = completion.choices[0].message.content
    message_history.append({"role": "user", "content": reply_content})

    response = [(message_history[i]["content"], message_history[i + 1]["content"]) for i in
                range(2, len(message_history) - 1, 2)]

    return response 

# Main function
with gr.Blocks(theme=gr.themes.base) as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        text_box = gr.Textbox(show_label=False, placeholder="Type your message here").style(container=False)
        text_box.submit(chatbot_response, text_box, chatbot)
        text_box.submit(lambda: "", None, text_box)

demo.launch()

