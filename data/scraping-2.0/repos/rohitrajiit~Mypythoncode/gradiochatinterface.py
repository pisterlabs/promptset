import gradio as gr
from openai import OpenAI

api_key = "sk-"  # Replace with your key

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})


    client = OpenAI(
    api_key=api_key,)
    response = client.chat.completions.create(
    messages=history_openai_format,
    # model="gpt-3.5-turbo" # gpt 3.5 turbo
    # model="gpt-4",
    model = "gpt-4-1106-preview", #gpt-4 turbo
    stream = True
    )

    partial_message = ""
   
    for chunk in response:
        text = (chunk.choices[0].delta.content)
        if text is not None:
            partial_message = partial_message + text
            yield partial_message

gr.ChatInterface(predict).queue().launch()
