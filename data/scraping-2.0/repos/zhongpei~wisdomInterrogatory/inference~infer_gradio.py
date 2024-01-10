import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import re
import torch
import gradio as gr
import sys
sys.path.append("/root/data1/luwen/app/langchain_demo/code")
from clc.langchain_application import LangChainApplication
from transformers import StoppingCriteriaList, StoppingCriteriaList
from clc.callbacks import Iteratorize, Stream
from langchain.schema import Document

class LangChainCFG:
    llm_model_name = 'luwen_baichuan/output/zju_model_0818_110k'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'app/langchain_demo/model/text2vec'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = 'app/langchain_demo/data/cache/legal_articles'
    kg_vector_stores = {
        '法律法条': 'app/langchain_demo/data/cache/legal_articles',
        '法律书籍': 'app/langchain_demo/data/cache/legal_books',
        '法律文书模板':'app/langchain_demo/data/cache/legal_templates',
        '法律案例': 'app/langchain_demo/data/cache/legal_cases',
        '法律考试': 'app/langchain_demo/data/cache/judicialExamination',
        '日常法律问答': 'app/langchain_demo/data/cache/legal_QA',
    }  

config = LangChainCFG()
application = LangChainApplication(config)

def clear_session():
    return '', None, ""

def predict(input,
            kg_names=None,
            history=None,
            intention_reg=None,
            **kwargs):
    large_language_model="zju-bc"
    max_length=1024
    top_k = 1
    application.llm_service.max_token = max_length
    # print(input)
    if history == None:
        history = []
    search_text = ''

    now_input = input
    eos_token_ids = [application.llm_service.tokenizer.eos_token_id]
    application.llm_service.history = history[-5:]
    max_memory = 4096 - max_length

    if len(history) != 0:
        if large_language_model=="zju-bc":
            input = "".join(["</s>Human:\n" + i[0] +"\n" + "</s>Assistant:\n" + i[1] + "\n"for i in application.llm_service.history]) + \
            "</s>Human:\n" + input
            input = input[len("</s>Human:\n"):]
        else:
            input = "".join(["### Instruction:\n" + i[0] +"\n" + "### Response: " + i[1] + "\n" for i in application.llm_service.history]) + \
            "### Instruction:\n" + input
            input = input[len("### Instruction:\n"):]
    if len(input) > max_memory:
        input = input[-max_memory:]

    print("histroy in call: ", history)
    prompt = application.llm_service.generate_prompt(input, kb_based, large_language_model)
    print("prompt: ",prompt)
    inputs = application.llm_service.tokenizer(prompt, return_tensors="pt").to('cuda')
    stopping_criteria = StoppingCriteriaList()

    kwargs['inputs'] = inputs
    kwargs['max_new_tokens'] = max_length
    kwargs['repetition_penalty'] = float(1.2)
    kwargs['stopping_criteria'] = stopping_criteria
    history.append((now_input, ""))

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        with torch.no_grad():
            application.llm_service.model.generate(**kwargs['inputs'], 
                                                    max_new_tokens=kwargs['max_new_tokens'], 
                                                    repetition_penalty=kwargs['repetition_penalty'],
                                                    stopping_criteria=kwargs["stopping_criteria"])

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**kwargs) as generator:
        for output in generator:
            last = output[-1]
            output = application.llm_service.tokenizer.decode(output, skip_special_tokens=True)
            pattern = r"\n{5,}$"
            pattern2 = r"\s{5,}$"
            origin_output = output
            if large_language_model=="zju-bc":
                output = output.split("Assistant:")[-1].strip()
            else:
                output = output.split("### Response:")[-1].strip()
            history[-1] = (now_input, output)
            yield "", history, history
            if last in eos_token_ids or re.search(pattern, origin_output) or re.search(pattern2, origin_output):
                break

with gr.Blocks() as demo: 
    state = gr.State()
    # with gr.Row():
    with gr.Column(scale=1):
        github_banner_path = 'https://raw.githubusercontent.com/LIANG-star177/chatgptapi/master/logo.png'
        gr.HTML(f'<p align="center"><a href="https://github.com/LIANG-star177/chatgptapi/blob/master/logo.png"><img src={github_banner_path} height="100" width="200"/></a></p>')

        with gr.Column(scale=4):
            with gr.Row():
                chatbot = gr.Chatbot(label='智海-录问').style(height=500)            
            with gr.Row():
                message = gr.Textbox(label='请输入问题')            
            with gr.Row():
                clear_history = gr.Button("🧹 清除历史对话")
                send = gr.Button("🚀 发送")
            with gr.Row():
                gr.Markdown("""<center>powered by 浙江大学 阿里巴巴达摩院 华院计算</center>""")

        send.click(predict,
                   inputs=[
                    message,
                    state,
                   ],
                   outputs=[message, chatbot, state],
                   show_progress=True)

        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[chatbot, state],
                            queue=False)

        message.submit(predict,
                       inputs=[
                        message,
                        state,
                       ],
                       outputs=[message, chatbot, state],
                       show_progress=True)

demo.queue(concurrency_count=2).launch(
    server_name='0.0.0.0',
    server_port=7888,
    share=True,
    enable_queue=True,
    inbrowser=True,
)
