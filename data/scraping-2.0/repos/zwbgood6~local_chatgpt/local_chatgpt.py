import os
import openai
import gradio as gr

# step 1: Run the following command to create a new environment variable named OPENAI_API_KEY:
# export OPENAI_API_KEY=<your-api-key>
# step 2: Save the changes to your environment variables by running the following command:
# source ~/.bashrc
# step 3: You can now check if the environment variable has been set by running the following command:
# echo $OPENAI_API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']

messages = [
    {"role": "system", 
     "content": "You are a helpful AI assistant."}, 
]

def chat(user_input):
    if user_input:
        messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(label="User input")
outputs = gr.outputs.Textbox(label="Response")

gr.Interface(
    fn=chat, 
    inputs=inputs, 
    outputs=outputs, 
    title="ChatGPT Demo",
    ).launch(share=True)
