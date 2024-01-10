import gradio as gr
from openai import OpenAI

api_key = "sk-"  # Replace with your key

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    filename = gr.File()

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(msg, history,filename):
        history_openai_format = []
        with open(filename, 'r') as file:
            # Read the contents of the file
            file_contents = file.read()
            history_openai_format.append({"role": "user", "content": file_contents })
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human })
            if assistant is not None:
                history_openai_format.append({"role": "assistant", "content":assistant})
        
        client = OpenAI(
        api_key=api_key,)
        response = client.chat.completions.create(
        messages=history_openai_format,
        # model="gpt-3.5-turbo" # gpt 3.5 turbo
        # model="gpt-4",
        model = "gpt-4-1106-preview", #gpt-4 turbo
        stream = True
        )
        history[-1][1] = ""
        partial_message = ""
        for chunk in response:
            text = (chunk.choices[0].delta.content)
            if text is not None:
                for character in text:
                        history[-1][1] += character
                        yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [msg, chatbot,filename], chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch()
