"""
基于grad io进行应用构建
https://github.com/gradio-app/gradio/issues/3729
"""
import logging
import os
import pathlib
import pickle
import shutil
import sys

import gradio as gr
import openai

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.insert(0, os.path.split(rootPath)[0])

from src.utils.embedding import get_embedding
from web import host, port, office_model_name, office_openai_key, api_version, api_base, api_type, azure_model_name, \
    azure_openai_key
from data import example, prompt_text
from src.gpt import set_openai_key, GPT, Example
from src.utils.data_store import doc2embedding, save_embedding
from src.utils.doc import parser_doc, hashcode_with_file, get_file_ext_size
from src.extract import parser_pdf, extract_doc, chat_mem_fin_llm

model_type = 'openai'

data_store_base_path = 'data/store'  # 生成文件父级目录
store_origin_file_dir = None
gpt: GPT = None
mem_api_base = os.getenv('MEM_FIN_OPENAI_API')


def load_model(model_type):
    """
    根据模型类型加载不同参数
    """
    global gpt

    # 使用同一实例，可以考虑单例模式进行优化
    if gpt is not None and openai.api_type == model_type:
        logging.info(f'reload {openai.api_type}')
        return gpt
    elif model_type == 'azure':
        set_openai_key(azure_openai_key, api_version, api_base, api_type)
        gpt = GPT(engine=azure_model_name, temperature=0.6, max_tokens=1024)
        return gpt

    elif model_type == 'open_ai':
        set_openai_key(office_openai_key)
        gpt = GPT(engine=office_model_name, temperature=0.6, max_tokens=1024)
        return gpt
    else:
        logging.info(model_type)


def init_store_dir(store_dir):
    """
    初始化app时 删除数据存储目录
    Returns: None

    """
    for root, dirs, files in os.walk(store_dir):
        for item in dirs:
            dir_path = os.path.join(root, item)
            shutil.rmtree(dir_path)
    logging.info("Clean store data.")


def process_upload_file(file_tmp_path: list):
    """
    Receive upload file and process.
    Args:
        file_tmp_path:

    Returns:str  file_name

    """
    global store_origin_file_dir
    file_name_path = file_tmp_path.name
    doc_name_with_ext = file_name_path.split('/')[-1]  # 上传文件的名称
    ext, file_size = get_file_ext_size(file_name_path)

    if str.lower(ext) not in ['.pdf', '.txt', '.docx', '.doc'] or file_size / 1024 / 1024 > 2:
        raise gr.Error(f"{ext} 请确认文件格式和大小 {file_size / 1024 / 1024}M")

    file_hashcode = hashcode_with_file(file_name_path)
    store_origin_file_dir = f'{data_store_base_path}/{file_hashcode}'
    logging.info(store_origin_file_dir)

    if pathlib.Path(store_origin_file_dir).exists():  # 存在表示文件已经已经上传过，不在继续后续逻辑
        logging.info("upload file exists.")
    else:
        pathlib.Path(store_origin_file_dir).mkdir(parents=True, exist_ok=True)
        copy_upload_file = f'{store_origin_file_dir}/{doc_name_with_ext}'
        shutil.copyfile(file_name_path, copy_upload_file)
        output_text_file = parser_doc(copy_upload_file, store_origin_file_dir)  # 统一解析输出为.txt
        embedding_with_index: dict = doc2embedding(output_text_file)  # 根据openai或其他embedding服务将句子转化为词向量
        save_embedding(embedding_with_index, f'{store_origin_file_dir}/embedding.pickle')  # 存储至本地

    return f'{doc_name_with_ext}预处理完成。'


def chat_doc(query, model_type, task_type='问答'):
    import numpy as np
    # Load knowledge from store
    try:
        if not store_origin_file_dir:
            logging.warning("Not found doc vector file.")
            return "无doc信息，考虑上传一份文档后再提问。"
        file = open(f'{store_origin_file_dir}/embedding.pickle', 'rb')
        emb_data = pickle.load(file)
        index, data = emb_data['index'], emb_data['embedding']
        logging.info("Success load doc vector file. Start query embedding……")

        emb, query_token_num = get_embedding(query)  # compute query embedding
        logging.info(f"query token num:{query_token_num}")
        _, text_index = index.search(np.array([emb[0][1]]), k=15)  # 根据索引从上传文档中搜索相近的内容
        context = []
        for i in list(text_index[0]):
            context.extend(data[i:i + 6])
        lens = [len(text) for text in context]
        logging.debug(f"匹配到的文本长度大小：{lens}")
        maximum = 3000
        for index, l in enumerate(lens):
            maximum -= l
            if maximum < 0:
                context = context[:index + 1]
                logging.warning("超过最大长度，截断到前", index + 1, "个片段")
                break

        text = "".join(text for _, text in enumerate(context))
        logging.info(f'Load model {model_type}')
        if model_type in ['azure', 'open_ai']:
            gpt = load_model(model_type)
            ret, tokens_num = gpt.get_top_reply(query, task_type, text, model_type)  # 请求LLM
            logging.info(f'Context:{text}\nOutput:{ret}')
            logging.info(f"本轮对话消耗tokens:{tokens_num}")
            return f'{model_type}\n{ret}'
        else:
            ret = chat_mem_fin_llm(mem_api_base, text, task_type)
            logging.debug(f'Context:{text}\nOutput:{ret}')
            return f'【{model_type}】\n{ret}'
    except Exception as e:
        logging.error(e)


def add_examples(issue, reply):
    """Generate QA example"""
    return gpt.add_example(Example(issue, reply))


def del_all_examples():
    [gpt.delete_example(ex_id) for ex_id, _ in gpt.get_all_examples().items()]
    return gpt.get_all_examples()


def task_with_chat(input_txt, task, model_type):
    """
    对话式任务
    Returns: response

    """
    logging.info(f'Load model name:{model_type}')
    try:
        logging.info(f'Query:{input_txt}')
        if model_type in ['open_ai', 'azure']:
            gpt = load_model(model_type)
            response, token_num = gpt.get_top_reply(input_txt, task, context='', model_type=model_type)
            logging.info(f"text len:{len(input_txt)}. Consumer token num:{token_num}. Response:{response}")
            return response
        elif model_type == 'all':
            gpt = load_model("azure")
            mem_response = chat_mem_fin_llm(mem_api_base, input_txt, task)
            gpt_response, token_num = gpt.get_top_reply(input_txt, task, context='', model_type='azure')
            return f'【MemectFinLLM】\n{mem_response} \n\n【gpt】\n{gpt_response}'

        else:
            response = chat_mem_fin_llm(mem_api_base, input_txt, task)
            logging.info(response)
            return response
    except Exception as e:
        gr.Error(e)


def extract_chain(file_path, schema, model_type):
    docs = parser_pdf(file_path)
    ret = extract_doc(schema, model_type, docs)
    return ret


with gr.Blocks(css="footer {visibility: hidden}", title='ChatLLM is all you need') as demo:
    gr.Markdown(f"<h1 style='text-align: center;'>大语言模型应用体验</h1>")
    gr.Markdown(f'> Model by MemFinLLM. Contact us via https://www.memect.cn/ .')

    with gr.Tab("场景问答"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="我要提问", placeholder="向大模型提问……")
                model_type = gr.Dropdown(choices=["memect", "openai", "azure", "all"], value='memect',
                                         label='选择模型类型')
                task_type = gr.Radio(choices=list(prompt_text.keys()),
                                     label="场景类型", value='问答')
                submit = gr.Button("问一下")
            with gr.Column():
                output_ret = gr.Text(label='输出', lines=8)
        submit.click(fn=task_with_chat, inputs=[input_text, task_type, model_type], outputs=output_ret)
        gr.Examples(example, [input_text, task_type])

    with gr.Tab("MemChatDoc（文档问答）"):  # 根据文档进行提问
        def add_file(history, doc):
            history = history + [(process_upload_file(doc), None)]
            return history

        def add_text(history, inp):
            history = history + [(inp, None)]
            return history, ""

        def bot(history, model_type):
            history[-1][1] = chat_doc(query=history[-1][0], model_type=model_type)
            return history

        chatbot = gr.Chatbot([("Welcome MemChatDoc. Please upload doc.", None)], show_label=False,
                             elem_id='chatbot').style(height="100%")
        model_type = gr.Dropdown(choices=["memect", "openai", "azure"], value='memect', label='选择模型类型')
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.85):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder='Enter question and press enter, or upload an file'
                ).style(container=False)

            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="📁上传文档", file_types=['file'])

        txt.submit(add_text, inputs=[chatbot, txt], outputs=[chatbot, txt], queue=False).then(bot,
                                                                                              [chatbot, model_type],
                                                                                              chatbot)
        btn.upload(add_file, inputs=[chatbot, btn], outputs=[chatbot]).then(bot, [chatbot, model_type], chatbot)

        clear = gr.Button("Clear")
        clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Tab("增加模型知识"):
        with gr.Column():  # 列排列
            question = gr.Textbox(label="question", value="文因互联是做什么的?")
            answer = gr.Textbox(label="answer",
                                value="文因互联是国内领先的金融认知智能解决方案的提供商，在知识图谱技术、自然语言处理技术、金融知识建模等方面有深厚积淀。")
            result = gr.Textbox(label='result')
        submit_example = gr.Button("submit_example")
        clean_example = gr.Button("clean_example")
        submit_example.click(fn=add_examples, inputs=[question, answer], outputs=result)
        clean_example.click(del_all_examples, inputs=[], outputs=result)

init_store_dir(data_store_base_path)
demo.launch(server_name=host, server_port=int(port), share=False)