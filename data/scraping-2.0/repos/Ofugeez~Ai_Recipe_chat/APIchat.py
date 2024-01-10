import os
import openai
import gradio as gr
from api_key import openai_api_key  # Import only the API key variable

openai.api_key = openai_api_key  # Set the API key using the imported variable


start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt="Hi and welcome",

def openai_create(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].text

def conversation_history(input, history):
    history = history or []
    s = list(sum(history,()))
    s.append(input)
    inp = ' '.join(s)
    output = openai_create(inp)
    history.append((input, output))
    return history, history

blocks = gr.Blocks()

with blocks:
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder=prompt)
    state = gr.State()
    submit = gr.Button("Click")
    submit.click(conversation_history, inputs=[message,state], outputs=[chatbot,state])

    blocks.launch(debug=True)