import os
import gradio as gr
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from keys import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 设置端口号，默认7560，遇冲突可自定义
SERVER_PORT = 7560


# 封装了predict 和 清除memory的接口调用
class myBot():
    def __init__(self):
        # 找到自定义的数据库
        loader = UnstructuredWordDocumentLoader('my_data/18 tourist.docx')
        data = loader.load()

        # 对数据进行分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

        # 准备模型和memory
        llm = ChatOpenAI()

        # 设置return_messages=True和 chat_history都是为了匹配输入格式
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

        retriever = vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=self.memory, verbose=True)

    def predict(self, input):
        return self.qa(input)

    def clear_memory(self):
        print(self.memory)
        self.memory.clear()
        print("正在清除memory")
        print(self.memory)


myGPT = myBot()


def predict(input, chatbot):
    chatbot.append((input, ""))
    response = myGPT.predict(input)
    chatbot[-1] = (input, response['answer'])
    return chatbot


def reset_user_input():
    return gr.update(value='')


def reset_state():
    myGPT.clear_memory()
    return []


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">{}</h1>""".format("MyBot"))
        # gradio的chatbot
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=50):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
        # 提交问题
        submitBtn.click(predict, [user_input, chatbot],
                        [chatbot])
        submitBtn.click(reset_user_input, [], [user_input])
        # 清空历史对话
        emptyBtn.click(reset_state, outputs=[chatbot])

    demo.queue().launch(share=False, inbrowser=True, server_port=SERVER_PORT)


if __name__ == '__main__':
    main()
