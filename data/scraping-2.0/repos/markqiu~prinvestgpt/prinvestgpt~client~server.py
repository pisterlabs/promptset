import itertools
import logging
import os
import time
from typing import Union

import PyPDF2
import gradio as gr
import gradio_user_history as gr_user_history
import meilisearch
import pandas as pd
import typer
from duckduckgo_search import DDGS
from langchain.chains import ConversationChain, RetrievalQA
from langchain.chat_models import ErnieBotChat
from langchain.embeddings import ErnieEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Meilisearch
from tqdm import tqdm

from prinvestgpt import settings

app = typer.Typer()

llm_model_config = settings.llm_model_config
embedding_model_config = settings.embedding_model_config
ernie_client_id = settings.ernie_client_id
ernie_client_secret = settings.ernie_client_secret
openai_api_key = settings.openai_api_key
meilisearch_url = settings.meilisearch_url
meilisearch_api_key = settings.meilisearch_api_key


# => Inject gr.OAuthProfile
def generate(prompt: str, profile: Union[gr.OAuthProfile, None]):
    image = ...

    # => Save generated image(s)
    gr_user_history.save_image(label=prompt, image=image, profile=profile)
    return image


# llm
def ddg_search(to_search):
    web_content = ""
    count = 1
    with DDGS(timeout=10) as ddgs:
        answer = itertools.islice(ddgs.text(f"{to_search}", region="cn-zh"), 5)  #
        for result in answer:
            web_content += f"{count}. {result['body']}"
            count += 1
        # instant = itertools.islice(ddgs.answers(f"kingsoft"), 5)#,region="cn-zh"
        # for result in instant:
        #     web_content += f"{count}. {result['text']}\n"
    return web_content


def pre_embedding_file(chat_bot):
    return [*chat_bot, ["预热知识库中，请耐心等待完成......", None]]


def apply_data(chat_bot):
    return [*chat_bot, ["载入知识库成功", None]]


def is_use_database(chat_bot, use_database_str):
    if use_database_str == "是":
        msg = "使用知识库中...."
    else:
        msg = "取消使用知识库"
    return [*chat_bot, [msg, None]]


def apply_model_setting(model_name, embedding_model, chat_bot):
    msg = f"载入语言模型{model_name}，embedding模型{embedding_model}"
    return [*chat_bot, [msg, None]]


def init_model(llm_model_name, embedding_model_name, temperature_value, max_tokens):
    llm_model = ErnieBotChat(
        ernie_client_id=ernie_client_id,
        ernie_client_secret=ernie_client_secret,
        temperature=temperature_value,
    )

    embedding_model = ErnieEmbeddings(
        ernie_client_id=ernie_client_id,
        ernie_client_secret=ernie_client_secret,
    )
    return llm_model, embedding_model


def general_template(history_flag=None):
    prompt_str = """请以一个资深投资AI的角色与人类的对话. The AI provides lots of specific
    details from its context. 如果AI不知道问题的答案，AI会诚实地说"我不知道"，而不是编造一个答案。
    AI在回答问题会注意自己的身份和角度。
----
Current conversation:"""
    if history_flag:
        prompt_str += """
{history}"""
        prompt_str += """
Human: {input}
AI: """
    else:
        prompt_str += """
已知内容：
'''{context}'''
"""
        prompt_str += """
Human: {question}
AI: """
    return prompt_str


def init_base_chain(llm_model_class, history_flag=None, user_question=None):
    template = general_template(history_flag=True)
    chain = ConversationChain(
        llm=llm_model_class,
        verbose=True,
        memory=history_flag,
    )
    chain.prompt.template = template
    try:
        output = chain.run(user_question)
    except Exception as e:
        raise e
    return output


def init_base_embedding_chain(llm_model_class, embedding_model_class, knowledge_database, user_question):
    if knowledge_database:
        template = general_template()
        qa_chain_prompt = PromptTemplate.from_template(template)
        vector_db = meilisearch.Client(url=meilisearch_url, api_key=meilisearch_api_key)
        vector_store = Meilisearch(
            embedding=embedding_model_class,
            client=vector_db,
            index_name="jinniuai",
            text_key="text",
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model_class,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": qa_chain_prompt, "verbose": True},
        )
        try:
            output = qa_chain.run(user_question)
        except Exception as e:
            raise e
        return output


def sheet_to_string(sheet):
    result = []
    for _index, row in sheet.iterrows():
        row_string = ""
        for column in sheet.columns:
            row_string += f"{column}: {row[column]}, "
        row_string = row_string.rstrip(", ")
        row_string += "."
        result.append(row_string)
    return result


def excel_to_string(file_path):
    # 读取Excel文件中的所有工作表
    excel_file = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)

    # 初始化结果字符串
    result = []

    # 遍历每一个工作表
    for _sheet_name, sheet_data in excel_file.items():
        # 处理当前工作表并添加到结果字符串
        result += sheet_to_string(sheet_data)

    return result


def get_documents(file_src):
    from langchain.schema import Document
    from langchain.text_splitter import TokenTextSplitter

    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=30)

    documents = []
    for file_obj in file_src:
        filepath = file_obj.name
        filename = os.path.basename(filepath)
        file_type = os.path.splitext(filename)[1]
        try:
            if file_type == ".pdf":
                pdf_text = ""
                with open(filepath, "rb") as pdf_file_obj:
                    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
                    for page in tqdm(pdf_reader.pages):
                        pdf_text += page.extract_text()
                texts = [Document(page_content=pdf_text, metadata={"source": filepath})]
            elif file_type == ".docx":
                from langchain.document_loaders import UnstructuredWordDocumentLoader

                loader = UnstructuredWordDocumentLoader(filepath)
                texts = loader.load()
            elif file_type == ".pptx":
                from langchain.document_loaders import UnstructuredPowerPointLoader

                loader = UnstructuredPowerPointLoader(filepath)
                texts = loader.load()
            elif file_type == ".epub":
                from langchain.document_loaders import UnstructuredEPubLoader

                loader = UnstructuredEPubLoader(filepath)
                texts = loader.load()
            elif file_type == ".xlsx":
                text_list = excel_to_string(filepath)
                texts = []
                for elem in text_list:
                    texts.append(Document(page_content=elem, metadata={"source": filepath}))
            else:
                from langchain.document_loaders import TextLoader

                loader = TextLoader(filepath, "utf8")
                texts = loader.load()
        except Exception as e:
            raise e

        texts = text_splitter.split_documents(texts)
        documents.extend(texts)
    return documents


def load_embedding_chain(file_obj=None, url=None, embedding_model=None):
    if embedding_model is None:
        llm_model, embedding_model = init_model(
            llm_model_name=None,
            embedding_model_name=None,
            temperature_value=0.7,
            max_tokens=2000,
        )
    if file_obj:
        filepath = file_obj.name
    elif url:
        filepath = url
    else:
        msg = "请上传文件或者填写url"
        raise ValueError(msg)
    logging.info(filepath)
    # book_name = f"temp{secrets.randbelow(100000)}"
    book_name = "jinniuai"
    docs = get_documents([url if url else file_obj])
    vector_db = Meilisearch.from_documents(
        docs,
        embedding_model,
        client=meilisearch.Client(url=meilisearch_url, api_key=meilisearch_api_key),
        index_name="jinniuai",
    )
    return vector_db, book_name


# gradio
block = gr.Blocks(css="footer {visibility: hidden}", title="文言一心助手")
with block:
    history = ConversationBufferMemory()
    history_state = gr.State(history)  # 历史记录的状态
    llm_model_state = gr.State()  # llm模型的状态
    embedding_model_state = gr.State()  # embedding模型的状态
    milvus_books = None
    milvus_books_state = gr.State(milvus_books)  # milvus_books的状态
    trash = gr.State()  # 垃圾桶

    with gr.Row():
        # 设置行

        with gr.Column(scale=1):
            with gr.Accordion("模型配置", open=False):
                llm_model_name = gr.Dropdown(
                    choices=llm_model_config,
                    value=llm_model_config[0],
                    label="语言模型",
                    multiselect=False,
                    interactive=True,
                )
                embedding_model_name = gr.Dropdown(
                    choices=embedding_model_config,
                    value=embedding_model_config[0],
                    label="embedding模型",
                    multiselect=False,
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="temperature",
                    interactive=True,
                )
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=16384,
                    value=1000,
                    step=1,
                    label="max_tokens",
                    interactive=True,
                )
                modle_settings = gr.Button("应用")

            use_database = gr.Radio(["是", "否"], label="是否使用知识库", value="否")

            with gr.Accordion("知识库选项", open=False):
                with gr.Tab("上传"):
                    file = gr.File(
                        label="上传知识库文件",
                        file_types=[
                            ".txt",
                            ".md",
                            ".docx",
                            ".pdf",
                            ".pptx",
                            ".epub",
                            ".xlsx",
                        ],
                    )
                    init_dataset_upload = gr.Button("应用")
                with gr.Tab("链接载入"):
                    knowledge_url_box = gr.Textbox(
                        label="url载入知识库",
                        placeholder="请粘贴你的知识库url",
                        show_label=True,
                        lines=1,
                    )
                    init_dataset_url = gr.Button("应用")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="小原同学")
            with gr.Row():
                message = gr.Textbox(
                    label="在此处填写你的问题",
                    placeholder="我有投资想法...",
                    lines=1,
                )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                # 刷新
                clear = gr.Button("刷新", variant="secondary")


            def clear_():
                chat_bot = []
                return "", chat_bot, ConversationBufferMemory()


            def user(user_message, history_flag):
                return "", [*history_flag, [user_message, None]]


            def bot(
                    user_message,
                    chat_bot=None,
                    history_state_value=None,
                    temperature=None,
                    max_tokens=None,
                    llm_model=None,
                    embedding_model=None,
                    llm_model_name=None,
                    embedding_model_name=None,
                    use_database_flag=None,
                    milvus_books_state_class=None,
            ):
                try:
                    history_state_value = ConversationBufferMemory()
                    user_message = chat_bot[-1][0]
                    if llm_model is None or embedding_model is None:
                        llm_model, embedding_model = init_model(
                            llm_model_name,
                            embedding_model_name,
                            temperature,
                            max_tokens,
                        )
                    if use_database_flag == "否":
                        output = init_base_chain(
                            llm_model,
                            history_flag=history_state_value,
                            user_question=user_message,
                        )
                    else:
                        output = init_base_embedding_chain(
                            llm_model,
                            embedding_model,
                            milvus_books_state_class,
                            user_question=user_message,
                        )
                except Exception as e:
                    raise e
                chat_bot[-1][1] = ""
                for character in output:
                    chat_bot[-1][1] += character
                    time.sleep(0.03)
                    yield chat_bot

    # 是否使用知识库
    use_database.change(is_use_database, inputs=[chatbot, use_database], outputs=[chatbot])
    # 模型配置
    modle_settings.click(
        init_model,
        inputs=[llm_model_name, embedding_model_name, temperature, max_tokens],
        outputs=[llm_model_state, embedding_model_state],
    ).then(
        apply_model_setting,
        inputs=[llm_model_name, embedding_model_name, chatbot],
        outputs=[chatbot],
    )
    # 知识库选项
    init_dataset_upload.click(pre_embedding_file, inputs=[chatbot], outputs=[chatbot]).then(
        load_embedding_chain,
        inputs=[file, embedding_model_state],
        outputs=[trash, milvus_books_state],
    ).then(apply_data, inputs=[chatbot], outputs=[chatbot])
    init_dataset_url.click(pre_embedding_file, inputs=[chatbot], outputs=[chatbot]).then(
        load_embedding_chain,
        inputs=[knowledge_url_box, embedding_model_state],
        outputs=[trash, milvus_books_state],
    ).then(apply_data, inputs=[chatbot], outputs=[chatbot])

    # 刷新按钮
    clear.click(clear_, inputs=[], outputs=[message, chatbot, history_state])
    # send按钮
    submit.click(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot,
        [
            message,
            chatbot,
            history_state,
            temperature,
            max_tokens,
            llm_model_state,
            embedding_model_state,
            llm_model_name,
            embedding_model_name,
            use_database,
            milvus_books_state,
        ],
        [chatbot],
    )
    # 回车
    message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
        bot,
        [
            message,
            chatbot,
            history_state,
            temperature,
            max_tokens,
            llm_model_state,
            embedding_model_state,
            llm_model_name,
            embedding_model_name,
            use_database,
            milvus_books_state,
        ],
        [chatbot],
    )

    with gr.Accordion("历史对话", open=False):
        gr_user_history.render()


@app.command()
def start():
    # 启动参数
    block.queue(concurrency_count=settings.concurrency_count).launch(
        debug=settings.debug,
        server_name=str(settings.bind_address),
        server_port=settings.server_port,
        root_path=settings.root_path,
    )
