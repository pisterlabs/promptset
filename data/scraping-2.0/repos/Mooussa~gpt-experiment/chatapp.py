import gradio as gr
import openai

# Put your OpenAI API key below (instead of key.txt)
openai.api_key = open("key.txt", "r").read().strip()

# This for prompting it to tell a joke
#message_history = [{"role": "user", "content": f"You are a joke"},
 #                  {"role":"assistant", "content": f"OK"}]
message_history = []

def predict(input):
    global message_history
    message_history.append({"role": "user", "content": input})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = message_history
    )

    reply_content = completion.choices[0].message.content
    print(reply_content)
    message_history.append({"role": "assistant", "content": reply_content})
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(0, len(message_history)-1, 2)]
    return response


# Build the gradio app
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here").style(container=False)
        txt.submit(predict, txt, chatbot)
        #txt.submit(lambda: "", None, txt) => do the same as below
        txt.submit(None, None, txt, _js="() =>{''}")


demo.launch(share=True)
