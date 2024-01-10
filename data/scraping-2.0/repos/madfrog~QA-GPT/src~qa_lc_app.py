# coding=utf-8

import os
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from file_loader.text_loader import TextLoader
from file_loader.pdf_loader import PDFLoader
from indexer.index_builder import IndexBuilder
from text_splitter.text_splitter import TextSplitter
from retriever.retriever_builder import RetrieverBuilder
from vector_store.vector_store import VectorStoreWrapper
from tools.logging_helper import LoggerHelper
import tempfile

tempfile._TemporaryFileWrapper

import gradio as gr

qa_file_path = "./qa_files"
qa_chain = None
logger = LoggerHelper().get_logger()

def get_txt_loader(file_path):
    # split file name and its dir path
    temp_arr = str.split(file_path, "/")
    file_name = temp_arr[len(temp_arr) - 1]
    del temp_arr[len(temp_arr) - 1]
    folder_path = os.path.join(*temp_arr)
    folder_path.strip()
    folder_path = f'/{folder_path}'
    logger.debug(f'Get folder path: {folder_path}, file name: {file_name}')

    loader = DirectoryLoader(folder_path, glob=f"{file_name}") 
    # docs = loader.load()
    return loader


def get_pdf_loader(file_path):
    pdf_load = PDFLoader(file_path)
    return pdf_load.build_loader()


def build_index(loader):
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index


def qa_on_index(index, question):
    return index.query(question)


def init_qa_engine():
    # init llm
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    # file_loader = TextLoader(qa_file_path)
    #index_builder = IndexBuilder()
    #index = index_builder.build_index(loader) 

    global text_splitter
    text_splitter = TextSplitter()
    global vector_store_wrapper
    vector_store_wrapper = VectorStoreWrapper()
    global retriever_builder
    retriever_builder = RetrieverBuilder()

    vector_store = vector_store_wrapper.init_vector_store()
    multi_query_retriever = retriever_builder.build_retriever(vector_store)

    global qa_chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=multi_query_retriever)


def parsing_files(file_objs):
    for f in file_objs:
        logger.debug(f'uploaded file name: {f.name}')
        # check file type by its suffix
        loader = None
        file_type = 'txt'
        if f.name.endswith('.pdf'):
            logger.debug(f"{f.name} type is pdf")
            loader = get_pdf_loader(f.name)
            file_type = 'pdf'
        elif f.name.endswith('.md'):
            logger.debug(f"{f.name} type is markdown")
            loader = get_txt_loader(f.name)
            file_type = 'md'
        else:
            # default file type is txt
            logger.debug(f"{f.name} type is txt")
            loader = get_txt_loader(f.name)
        splitted_docs = text_splitter.split_text(loader.load(), file_type)
        vector_store_wrapper.add_docs_to_vector_store(splitted_docs)
        logger.info(f'add docs of {f.name} to vector store successfully.')
    return '解析成功'


def chatbot(input_text):
    # question = "华为云的计费方式是什么样的？"
    # question = "什么是遗留资产?"
    response = qa_chain({"query": input_text})
    print(response)
    return response['result']


if __name__=="__main__":
    init_qa_engine()
    app = gr.Blocks()
    with app:
        with gr.Tab('Upload docs'):
            files = gr.File(file_count="multiple", file_types=["text", ".json", ".md", ".pdf"], type="file", label='files')
            parsing_bnt = gr.Button('Begin to parse docs')
            upload_result = gr.HighlightedText(show_legend=False)

        with gr.Tab('Q&A'):
            with gr.Column():
                input_text = gr.Textbox(label='Qestion', lines=5)
                qa_bnt = gr.Button('Submit')
            with gr.Column():
                output_text = gr.Textbox(label='Answer', lines=5)

        parsing_bnt.click(parsing_files, inputs=files, outputs=upload_result)
        qa_bnt.click(chatbot, inputs=input_text, outputs=output_text)

    app.launch(share=True, server_name='0.0.0.0')
