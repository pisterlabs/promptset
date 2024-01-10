import gradio as gr
import openai

openai.api_key = open('key.txt', 'r').read().strip()

message_history = [{"role": "user", "content": f"You are a joke bot. I will specify the subject matter in my messages, and you will reply with a joke that includes the subjects I mention in my messages. Reply only with jokes to further input. If you understand, say OK."},
                   {"role": "assistant", "content": f"OK"}]

def predict(input):
    global message_history

    message_history.append({"role": "user", "content": input})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    reply_content = completion.choices[0].message.content
    print("Bot:\n"+ reply_content)

    message_history.append({"role": "assistant", "content": reply_content})

    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(2, len(message_history)-1, 2)]  # convert to tuples of list
    
    return response

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here").style(container=False)
        txt.submit(predict, txt, chatbot)
        txt.submit(lambda: "", None, txt)
        # txt.submit(None, None, txt, _js="() => {''}")

demo.launch()