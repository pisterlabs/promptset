import gradio as gr
import os
import json
import requests

# Streaming endpoint
API_URL = "https://api.openai.com/v1/chat/completions"  # os.getenv("API_URL") + "/generate_stream"

# Testing with my Open AI Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def predict(inputs, top_p, temperature, chat_counter, chatbot=[], history=[]):
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": f"{inputs}"}],
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "stream": True,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    print(f"chat_counter - {chat_counter}")
    if chat_counter != 0:
        messages = []
        for data in chatbot:
            temp1 = {}
            temp1["role"] = "user"
            temp1["content"] = data[0]
            temp2 = {}
            temp2["role"] = "assistant"
            temp2["content"] = data[1]
            messages.append(temp1)
            messages.append(temp2)
        temp3 = {}
        temp3["role"] = "user"
        temp3["content"] = inputs
        messages.append(temp3)
        # messages
        payload = {
            "model": "gpt-4",
            "messages": messages,  # [{"role": "user", "content": f"{inputs}"}],
            "temperature": temperature,  # 1.0,
            "top_p": top_p,  # 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

    chat_counter += 1

    history.append(inputs)
    print(f"payload is - {payload}")
    # make a POST request to the API endpoint using the requests.post method, passing in stream=True
    response = requests.post(API_URL, headers=headers, json=payload, stream=True)
    print(f"response code - {response}")
    token_counter = 0
    partial_words = ""

    counter = 0
    for chunk in response.iter_lines():
        # Skipping first chunk
        if counter == 0:
            counter += 1
            continue
        # counter+=1
        # check whether each line is non-empty
        if chunk.decode():
            chunk = chunk.decode()
            # decode each line as response data is in bytes
            if len(chunk) > 12 and "content" in json.loads(chunk[6:])['choices'][0]['delta']:
                # if len(json.loads(chunk.decode()[6:])['choices'][0]["delta"]) == 0:
                #  break
                partial_words = partial_words + json.loads(chunk[6:])['choices'][0]["delta"]["content"]
                if token_counter == 0:
                    history.append(" " + partial_words)
                else:
                    history[-1] = partial_words
                chat = [(history[i], history[i + 1]) for i in
                        range(0, len(history) - 1, 2)]  # convert to tuples of list
                token_counter += 1
                yield chat, history, chat_counter, response  # resembles {chatbot: chat, state: history}


def reset_textbox():
    return gr.update(value='')


title = """<h1 align="center">🔥GPT4 with ChatCompletions API +🚀Gradio-Streaming</h1>"""
description = """Language models can be conditioned to act like dialogue agents through a conversational prompt that typically takes the form:
```
User: <utterance>
Assistant: <utterance>
User: <utterance>
Assistant: <utterance>
...
```
In this app, you can explore the outputs of a gpt-4 LLM.
"""

theme = gr.themes.Default(primary_hue="green")

with gr.Blocks(css="""#col_container { margin-left: auto; margin-right: auto;}
                #chatbot {height: 520px; overflow: auto;}""",
               theme=theme) as demo:
    gr.HTML(title)
    gr.HTML(
        """<h3 align="center">🔥This Huggingface Gradio Demo provides you full access to GPT4 API (4096 token limit). 🎉🥳🎉You don't need any OPENAI API key🙌</h1>""")
    gr.HTML(
        '''<center><a href="https://huggingface.co/spaces/ysharma/ChatGPT4?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space and run securely with your OpenAI API Key</center>''')
    with gr.Column(elem_id="col_container"):
        # GPT4 API Key is provided by Huggingface
        # openai_api_key = gr.Textbox(type='password', label="Enter only your GPT4 OpenAI API key here")
        chatbot = gr.Chatbot(elem_id='chatbot')  # c
        inputs = gr.Textbox(placeholder="Hi there!", label="Type an input and press Enter")  # t
        state = gr.State([])  # s
        with gr.Row():
            with gr.Column(scale=7):
                b1 = gr.Button().style(full_width=True)
            with gr.Column(scale=3):
                server_status_code = gr.Textbox(label="Status code from OpenAI server", )

        # inputs, top_p, temperature, top_k, repetition_penalty
        with gr.Accordion("Parameters", open=False):
            top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.05, interactive=True,
                              label="Top-p (nucleus sampling)", )
            temperature = gr.Slider(minimum=-0, maximum=5.0, value=1.0, step=0.1, interactive=True,
                                    label="Temperature", )
            # top_k = gr.Slider( minimum=1, maximum=50, value=4, step=1, interactive=True, label="Top-k",)
            # repetition_penalty = gr.Slider( minimum=0.1, maximum=3.0, value=1.03, step=0.01, interactive=True, label="Repetition Penalty", )
            chat_counter = gr.Number(value=0, visible=False, precision=0)

    inputs.submit(predict, [inputs, top_p, temperature, chat_counter, chatbot, state],
                  [chatbot, state, chat_counter, server_status_code], )  # openai_api_key
    b1.click(predict, [inputs, top_p, temperature, chat_counter, chatbot, state],
             [chatbot, state, chat_counter, server_status_code], )  # openai_api_key
    b1.click(reset_textbox, [], [inputs])
    inputs.submit(reset_textbox, [], [inputs])

    # gr.Markdown(description)
    demo.queue(max_size=20, concurrency_count=10).launch(debug=True)