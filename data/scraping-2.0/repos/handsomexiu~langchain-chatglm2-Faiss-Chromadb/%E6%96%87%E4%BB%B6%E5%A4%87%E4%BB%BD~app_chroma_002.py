
import os
from typing import List

import gradio as gr
import nltk
import sentence_transformers
# 这里不需要web搜索
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma

from chatllm import ChatLLM
from chinese_text_splitter import ChineseTextSplitter
from config import *

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")
                  ] + nltk.data.path

embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
VECTOR_STORE_PATH=VECTOR_STORE_PATH
num_gpus = num_gpus#GPU数量
init_llm = init_llm
init_embedding_model = init_embedding_model




class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None

    def init_model_config(
        self,
        large_language_model: str = init_llm,
        embedding_model: str = init_embedding_model,
    ):#上面括号里面的是参数

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model], )
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device=EMBEDDING_DEVICE,
            cache_folder=os.path.join(MODEL_CACHE_PATH,
                                      self.embeddings.model_name))
        self.llm = ChatLLM()
        if 'chatglm2' in large_language_model.lower():#所有字符串小写，这里这样写的目的是llm_model_dict是一个二重字典
            self.llm.model_type = 'chatglm2'
            self.llm.model_name_or_path = llm_model_dict['chatglm2'][
                large_language_model]
                #这里和上面的embedding需要修改config中对应的字典的内容：如果本地部署模型需要模型的本地路径
        self.llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)

    def init_knowledge_vector_store(self, file_obj):
        # 由于不同于单文件的格式，多文件的格式上传的是一个列表
        # 因此这里可以查看这里可以查看是不是一个列表，对于列表和单个文件采取不一样的处理方式
        if isinstance(file_obj, list):
            docs=[]
            for file in file_obj:
                doc=self.load_file(file.name)
                docs.extend(doc)#这里不同于append，extend是将列表中的元素添加到另一个列表中
        else:
            docs = self.load_file(file_obj.name)
        print("文档拆分成功")
        print("docs:      ",docs)
        print(docs[0].metadata)
        db = Chroma.from_documents(docs, self.embeddings,persist_directory='./vector_store/chromadb1')
        return db

    def get_knowledge_based_answer(self,
                                   query,
                                   max_length: int=5000,
                                   top_k: int = 6,
                                   history_len: int = 3,
                                   temperature: float = 0.01,
                                   top_p: float = 0.1,
                                   history=[]):
        self.llm.max_token = max_length
        # print(history)#这里是为了检测state 的内容，state作为参数传到了history中
        self.llm.temperature = temperature
        self.llm.top_p = top_p
        self.history_len = history_len
        self.top_k = top_k#用于向量数据库
        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

            已知内容:
            {context}

            问题:
            {question}"""
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm.history = history[
            -self.history_len:] if self.history_len > 0 else []
        vector_store = Chroma(persist_directory='./vector_store/chromadb1', embedding_function=self.embeddings)

        knowledge_chain = RetrievalQA.from_llm(# 检索问答链
            llm=self.llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": self.top_k}),
            prompt=prompt)
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        return result

    def load_file(self, filepath):
        if filepath.lower().endswith(".md"):
            # loader = UnstructuredFileLoader(filepath, mode="elements")
            loader = UnstructuredFileLoader(filepath)
            # docs = loader.load()
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        else:
            # loader = UnstructuredFileLoader(filepath, mode="elements")
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs# list

# 这个函数好像没有用到
def update_status(history, status):
    history = history + [[None, status]]
    print(status)
    return history


knowladge_based_chat_llm = KnowledgeBasedChatLLM()

# 这个用来初始化模型
def init_model():
    try:
        knowladge_based_chat_llm.init_model_config()
        knowladge_based_chat_llm.llm._call("你好")
        return "初始模型已成功加载，可以开始对话"
    except Exception as e:
        return "模型未成功重新加载，请点击重新加载模型"


# 文件内容清除
def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)
    ls = os.listdir(path_file)#这里是为了检查空文件夹
    for i in ls:
        f_path = os.path.join(path_file, i)
        os.rmdir(f_path)

def clear_session():
    # 除了清空对话之外,还希望可以清空向量数据库中的文件
    del_files(VECTOR_STORE_PATH)
    return '', None

# 初始化向量数据库
def init_vector_store(file_obj):
    # print('file:      ',file_obj)
    # print('file.name:      ',file_obj.name)
    vector_store = knowladge_based_chat_llm.init_knowledge_vector_store(
        file_obj)
    print('vector_store加载完成')

    return vector_store

# 用来预测
def predict(input,
            max_length,
            top_k,
            history_len,
            temperature,
            top_p,
            history=None):
    if history == None:
        history = []


    resp = knowladge_based_chat_llm.get_knowledge_based_answer(
        query=input,
        max_length=max_length,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history)
    history.append((input, resp['result']))
    return '', history, history


model_status = init_model()

if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:
        model_status = gr.State(model_status)
        with gr.Row():
            with gr.Column(scale=1):
                #这里不需要模型选择，模型在开始的时候就已经加载进去了
                model_argument = gr.Accordion("模型参数配置")
                with model_argument:

                    max_length = gr.Slider(2000,
                                      10000,
                                      value=5000,
                                      step=1000,
                                      label="max token",
                                      interactive=True)

                    top_k = gr.Slider(1,
                                      10,
                                      value=6,
                                      step=1,
                                      label="vector search top k",
                                      interactive=True)

                    history_len = gr.Slider(0,
                                            5,
                                            value=3,
                                            step=1,
                                            label="history len",
                                            interactive=True)

                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="temperature",
                                            interactive=True)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.9,
                                      step=0.1,
                                      label="top_p",
                                      interactive=True)

                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx', '.pdf'],
                               file_count='multiple',#这里可以上传多个文件
                               height=170)

                init_vs = gr.Button("知识库文件向量化")


            with gr.Column(scale=4):
                chatbot = gr.Chatbot([[None, model_status.value]],
                                     label='ChatLLM',height=500)
                message = gr.Textbox(label='请输入问题')
                state = gr.State()

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史对话及知识文件")
                    send = gr.Button("🚀 发送")


            init_vs.click(
                init_vector_store,
                show_progress=True,
                inputs=[file],
                outputs=[],
            )

            send.click(predict,
                       inputs=[
                           message, max_length, top_k, history_len, temperature,
                           top_p, state
                       ],# 这里的state也可以用chatbot
                       outputs=[message, chatbot, state])
            clear_history.click(fn=clear_session,
                                inputs=[],
                                outputs=[chatbot, state],
                                queue=False)

            message.submit(predict,
                           inputs=[
                               message, max_length, top_k, history_len,
                               temperature, top_p, state
                           ],
                           outputs=[message, chatbot, state])
    # 这里的state表示的是历史？——是的
    # 通过验证，gradio.state会存储历史对话，除非点击clear_history
    # chatbot好像存的也是历史对话，chatbot和state都可以用来存储历史对话
    # threads to consume the request
    # demo.queue(concurrency_count=3) \
    demo.launch(server_name='0.0.0.0', # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
                server_port=7860, # the port for listening
                show_api=False, # if display the api document
                share=True, # if register a public url
                inbrowser=False) # if browser would be open automatically
