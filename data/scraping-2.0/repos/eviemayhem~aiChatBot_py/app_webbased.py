import gradio as gr
import openai
from api import key as api_key


#API Key goes here, store it in a different variable that you then gitignore
openai.api_key = api_key


def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        #For Content, this is where you would use prompt engineering to "train" how you'd like the bot to respond
        history_openai_format.append({"role": "assistant", "content":'Your name is Violet. You are a kind and caring friend who is here to help.You have extensive knowledge in Microsoft Dynamics D365 F+O as well as PowerBI and Data Analytics. '})
    history_openai_format.append({"role": "user", "content": message})

#Chat gpt response settings
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages= history_openai_format,
        temperature=1.0,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            yield partial_message

gr.ChatInterface(fn=predict,theme=gr.themes.Monochrome,).queue().launch()