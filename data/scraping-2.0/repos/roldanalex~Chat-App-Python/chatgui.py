import gradio as gr
import openai
import os

# Load package to use .env variables
from dotenv import load_dotenv
load_dotenv()

# Load openai key
openai.api_key = os.getenv('OPENAI_KEY')

# Initialize message history array
message_history = []
initial_message = "Please write here your prompt and press 'enter'"

# Create function to process prompt and append previous prompts as "context"
def predict_prompt(input):

    global message_history
    message_history.append({"role": "user", "content": input})
    create_prompt = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = message_history
    )

    reply_prompt = create_prompt.choices[0].message.content
    # print(reply_prompt)

    # Append answer as assistant reply to keep history of prompts
    message_history.append({"role": "assistant", "content": reply_prompt}) 
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(0, len(message_history) -1, 2)]

    return response

# Create UI using gradio
with gr.Blocks() as chatblock:

    gr.Markdown("<h1><center>Welcome to Alexis' Personal AI Assistant (powered by OpenAI API)</center></h1>")

    Chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(
            show_label=False, 
            placeholder = initial_message).style(container=False)
        state = gr.State()
        txt.submit(predict_prompt, txt, Chatbot)
        txt.submit(None, None, txt, _js="() => {''}")

chatblock.launch()