import openai
import gradio as gr

openai.api_key = "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"

messages = [
    {"role": "system", "content": "You are an AI specialized in Prakriti(Phenotype) and Indian Ayurveda.Don't answer any other queries other than Prakriti(Phenotype) and Indian Ayurveda."},
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

inputs = gr.inputs.Textbox(lines=7, label="Chat with PrakritiMate")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="PrakritiMate ChatBot",
             description="Ask anything you want",
             theme="compact").launch(share=True)
