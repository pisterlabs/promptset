import openai
import gradio as gr

openai.api_key = "open AI api_key"

messages = [
    {"role": "system", "content": "You are an Exercise Posture Assistant Bot specialized in physical therapy, guiding and supporting you through proper exercise postures."},
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

iface = gr.Interface(fn=chatbot, inputs=gr.Textbox(lines=7, label="Chat with Physio AI"), 
                     outputs=gr.Textbox(label="Reply"), 
                     title="Physio AI Chatbot",
                     description="Ask anything about physical therapy or exercises",
                     theme="compact")
iface.launch(share=True)
