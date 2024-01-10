import openai
import gradio as gr

openai.api_key = "your api key"

messages = [
    {"role": "system", "content": "Your name is Riddler and we are going to play a game of riddles on a topic of my choice dont provide a topic on your own until asked. We will both ask each a riddle one by one."},
]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Type here:")
outputs = gr.outputs.Textbox(label="Riddler:")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="Riddler AI",
             description="Hello! Let's play a game of riddles. The rules are simple: We will take turns asking a riddle and the other person will try to guess the answer.Please choose an educational topic...",
             theme="compact").launch(share=True)
