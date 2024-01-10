import openai
import gradio as gr
openai.api_key = "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"

def greet(content):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}]
    )
    return completion.choices[0].message.content

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(share=True)