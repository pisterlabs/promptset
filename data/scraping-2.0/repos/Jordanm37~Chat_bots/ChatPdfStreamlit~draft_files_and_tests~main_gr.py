import gradio as gr
# from pdf_chat_bot import chat_bot_response
import time
import openai
import os
from dotenv import load_dotenv


load_dotenv('.env')


openai.api_key = os.getenv('OPENAI_API_KEY')


def chat_bot_response(user_content):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert Greek translator and transcription agent. Your role is to carefully examine the given text in Greek, identify semantic errors resulting from transcription, and then correct these errors. Use your knowledge of Greek language and syntax to ensure the accuracy and coherence of the transcriptions. Your task is crucial for the quality and usefulness of the transcriptions. You will be given Greek text that has been transcribed but may contain errors. Your job is to fix these errors to the best of your ability. Respond using markdown."},
            {"role": "user", "content": user_content},
        ],
    )
    return response["choices"][0]["message"]["content"]



def main():

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=250)
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            bot_message = chat_bot_response(message)  # Use your chatbot_completition function here
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        
        demo.launch()

if __name__ == '__main__':
    main()
    