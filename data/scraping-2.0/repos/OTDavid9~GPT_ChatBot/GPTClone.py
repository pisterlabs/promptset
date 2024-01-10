from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("API_KEY")

## 
import gradio as gr

from openai import OpenAI

client = OpenAI(
   
    api_key=api_key,
)


def gpt_output(input):
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input}
        
    ]

    response = client.chat.completions.create(
         model="gpt-3.5-turbo",
         messages=conversation,
         temperature=1,
         max_tokens=256,
         top_p=1,
         frequency_penalty=0,
         presence_penalty=0
   
    )
 
    return response.choices[0].message.content

def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ''.join(s)
    output = gpt_output(input)
    history.append((input, output))
    return history, history


block = gr.Blocks()

with block:
    gr.Markdown("""<h1><center>AGI AI Assistant <center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder="Enter your Query")
    state = gr.State()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])

block.launch(debug=False, share=True)
