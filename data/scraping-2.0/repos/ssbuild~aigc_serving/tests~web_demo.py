import openai

import gradio as gr
import mdtex2html

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.101.30:8081/v1"


import sys 
sys.path.append(".") 

from serving.config_loader.main import global_models_info_args

models = list()
for model_name,model_config in global_models_info_args.items():
    if model_config['enable']:
        models.append(model_name)


# # Test list models API
# models = openai.Model.list()
# print("Models:", models)

# Test completion API
stream = True
prefix = "你是蔚来汽车上的的人工智能助理，你的名字叫小诚。你在电动汽车领域非常专业，非常精通蔚来汽车的所有功能和服务。当问道你是谁？你要回答我是蔚来小诚。\n请你回答如下问题：\n"

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.build_args = postprocess

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def predict(input, chatbot, model, max_length, top_k, top_p, temperature, repetition_penalty, do_sample, history):
    chatbot.append((parse_text(input), ""))
       
    data = {
        "model": model,
        "adapter_name": None, # lora头
        "messages": [{"role": "system", "content": prefix } ,{"role": "user", "content": input} ],
        # "messages": [{"role": "user", "content": prefix + input}],
        "top_p": top_p,
        "temperature": temperature,
        "frequency_penalty": repetition_penalty,
        "stream": stream,
        "max_tokens": 512,
        "nchar": 1,# stream 字符
        "n": 1 # 返回 n 个choices
    }
    
    completion = openai.ChatCompletion.create(**data)
    if stream:
        response = model+':'
        for choices in completion:
            c = choices.choices[0]
            delta = c.delta
            if hasattr(delta,'content'):
                response += delta.content
                chatbot[-1] = (parse_text(input), parse_text(response))
                #print(delta.content)
                yield chatbot, history
    else:
        for choice in completion.choices:
            chatbot[-1] = (parse_text(input), parse_text(choice.message.content)) 
            #print(choice.message.content)
            yield chatbot, history
    

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">蔚来小诚</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=20).style(container=True)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            model = gr.Dropdown(choices=models, label="Model", value=models[0], type="value",interactive=True),
            max_length = gr.Slider(0, 2048, value=2048, step=128, label="Maximum length", interactive=True)
            top_k = gr.Slider(0, 30, value=5, step=1, label="Top K", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)
            repetition_penalty = gr.Slider(1, 1.5, value=1.05, step=0.01, label="Repetition penalty", interactive=True)
            do_sample = gr.Radio(["YES", "NO"], label="Do sample", type="index", value="YES")


    history = gr.State([])
    
    submitBtn.click(predict, [user_input, chatbot, model[0], max_length, top_k, top_p, temperature, repetition_penalty, do_sample, history], [chatbot, history], show_progress=True)
    
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, server_name="0.0.0.0", server_port=8001)
