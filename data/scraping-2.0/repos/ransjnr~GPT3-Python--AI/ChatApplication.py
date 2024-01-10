#GPT-3 powered AGI Chat Application: RANSFORD OPPONG: Aug 4,2023
import os
import openai
import gradio as gr
openai.api_key = "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"
#command to tell the model how to arrange the inputs and outputs.
start_sequence = "\nAI:"
restart_sequence = "\Human: "
#initial input
prompt ="The following is a conversation with an AI Assistant. The Assistant is helpful, creative, clever and very friendly. \n\nHuman: Hello, who are you\nAI: I am an AI created by OpenAI. How may I assist you today?\nHuman: ",
def gpt_output(prompt):
    response = openai.Completion.create(
        model ="text-davinci-003",
        prompt = prompt,
        temperature = 0.9,
        max_tokens = 150,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0.6,
        stop = ["Human: ","AI: "]
    )
    return response.choice[0].text
#a loop to take input from a user when the function is true
# while True:
#     query = input("Ask a QUestion to AI:\n")
#     gpt_output(query)

#context storage or history
def chatgpt_clone(input,history):
    history = history or []
    s = list(sum(history,()))
    s.append(input)
    inp = ''.join(s)
    output = gpt_output(inp)
    history.append((input,output))
    return history,history
#pip install gradio - chat application web interface
block = gr.Blocks()
#builtin gradio functions for the interface
with block:
    gr.Markdown("""<h1><center>Rans AI Assistant</center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox(placeholder = prompt)
    state = gr.State()
    # session = gr.File()
    submit = gr.Button("SEND")
    submit.click(chatgpt_clone,inputs=[message,state],outputs=[chatbot,state])
#set the launch to true
block.launch(debug=True)