#!/usr/bin/env python3

import os
import shutil
import sys
import tempfile
import threading
import time

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


model_name = "gpt-3.5-turbo"
temperature = 0.1
max_tokens = 2048
chunk_size = 500
similarity_search_k = 4
translateNeeded = True
translatedAnswer = ""


def parse_env_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = current_dir + "/env.config"
    if os.path.exists(config):
        with open(config, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("OPENAI_API_KEY"):
                    os.environ['OPENAI_API_KEY'] = line.split("=")[1].strip("\"\'\r\n")
    else:
        print("env.config not found!")
    if os.environ['OPENAI_API_KEY'] == '':
        print("OPENAI_API_KEY not found! Please config it in env.config")
        exit(1)
    

def get_vector_db(name: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "vector_db")
    if not os.path.exists(db_dir):
        os.mkdir(db_dir)
    return db_dir + "/" + name


def translate(txt):
    print("Translate to Chinese...")
    import openai
    openai.api_key = os.environ['OPENAI_API_KEY']
    completion = openai.ChatCompletion.create(model=model_name, messages=[
        {"role": "user", "content": "translate below content into Chinese"
                                    "(professional terminology and code do not need to be translated):" + txt}])
    result = completion.choices[0].message.content
    # print(result)
    return result

    # 下面的代码报错： Retrying langchain.chat_models.openai.acompletion_with_retry.<locals>._completion_with_retry in 8.0
    # seconds as it raised APIConnectionError: Error communicating with OpenAI.
    # llm = ChatOpenAI(temperature=0,max_tokens=1024)
    # g = await llm.agenerate([[HumanMessage(content="translate below content into Chinese:"+txt)]])
    # print(g)


def create_vectors(file_url: str):
    db_path = get_vector_db(file_url)
    if not os.path.exists(db_path):
        print("create new chroma store...")
        is_pdf = os.path.splitext(file_url)[1] == '.pdf'
        if is_pdf:
            # PDFMinerLoader是一个Python库，用于从PDF文档中提取文本。
            # 它使用先进的算法来准确地识别和提取PDF中的文本，包括图像中的文本。
            # 它还可以用于从PDF中提取元数据，例如作者、标题和主题。
            loader = PDFMinerLoader(file_url)
            file_data = loader.load()
        # TODO: 添加其他类型文件
        else:
            # 作为文本文件
            with open(file_url, 'r') as f:
                text = f.read()
            metadata = {"source": file_url}
            file_data = [Document(page_content=text, metadata=metadata)]

        # RecursiveCharacterTextSplitter类用于将文本分成较小的字符块。
        # chunk_size参数指定每个块的最大大小，chunk_overlap参数指定两个相邻块之间应重叠多少个字符。
        # chunk_size设置为N，chunk_overlap设置为0，这意味着每个块将有N个字符长，并且相邻块之间没有重叠。
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        split_documents = textsplitter.split_documents(file_data)
        print("split_documents:", len(split_documents))

        # OpenAIEmbeddings类用于将文本转换为向量表示。
        # 使用Chroma库来创建一组文档的向量表示。文档作为“split_documents”参数传入，使用的嵌入是OpenAIEmbeddings()。
        # persist_directory参数指定了存储向量表示的目录。这样可以重复使用向量表示，而无需每次重新创建它们。
        vectors_store = Chroma.from_documents(documents=split_documents, embedding=OpenAIEmbeddings(),
                                              persist_directory=db_path)
        vectors_store.persist()
        print('load finish!')
    else:
        print('has loaded!')


def ask_ai(file_url: str, question: str):
    db_path = get_vector_db(file_url)
    if not os.path.exists(db_path):
        create_vectors(file_url)
    else:
        print('use existing chroma store...')

    # 创建一个Chroma对象，使用OpenAIEmbeddings作为嵌入函数，并使用get_persist_path作为持久化目录。
    vectors_store = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=db_path)

    llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    system_template = """Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")]
    # ChatPromptTemplate可以用来创建一个聊天机器人提示模板，以便在聊天时使用。
    prompt = ChatPromptTemplate.from_messages(messages)
    # 链对象可以用于基于检索的问答任务。
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff",
                                                        retriever=vectors_store.as_retriever(),
                                                        return_source_documents=False,
                                                        chain_type_kwargs={"prompt": prompt})
    print("search similarity content in vectors....")
    # 检索相似“文档”（这里的“文档”是我们的文档被split后的子文档）
    similarity_doc = vectors_store.similarity_search(question, k=similarity_search_k)

    chain_params = {"input_documents": similarity_doc,
                    "question": question,
                    "return_only_outputs": True,}

    def show_busy_indicator():
        while chain_thread.is_alive():
            for c in '|/-\\':
                print('\r' + 'THINKING... ' + c, end='', flush=True)
                time.sleep(0.1)
        print("\r", end='', flush=True)

    def run_chain(params):
        chat_res = chain(params)
        answer = chat_res.get('answer')
        print("\n"+answer)
        global translatedAnswer
        if translateNeeded:
            translatedAnswer = translate(answer)
        else:
            translatedAnswer = answer

    # chat_res = chain({"input_documents": similarity_doc,"question": question}, return_only_outputs=True)
    chain_thread = threading.Thread(target=lambda: run_chain(chain_params))
    chain_thread.start()
    show_busy_indicator()
    chain_thread.join()
    # print as green
    print('\033[32m' + "---- answer ---- " + '\033[0m')
    global translatedAnswer
    if translateNeeded:
        print(translatedAnswer)


def main():
    parse_env_config()

    # 使用方式1：使用stdin作为内容输入
    # cat 123.txt | python3 chatdoc.py question
    if not sys.stdin.isatty():
        # remove all  ./vector_db/var/*
        temp_dir = get_vector_db("var")
        if os.path.exists(temp_dir):
            # print(temp_dir)
            shutil.rmtree(temp_dir)

        std = ""
        # read all lines from stdin
        for line in sys.stdin:
            std += line
        # create a temp file to save the content
        temp_file = tempfile.NamedTemporaryFile(delete=True)
        temp_file.write(std.encode('utf-8'))
        # combine all arguments into a single string
        combined_question = " ".join(sys.argv[1:])
        ask_ai(temp_file.name, combined_question)
        temp_file.close()
        return 0

    # 使用方式2：使用文件作为内容输入
    # python3 chatdoc.py file_url question
    if len(sys.argv) < 3:
        print('Usage: python3 chatdoc.py file_url question')
        sys.exit(1)
    file_url = sys.argv[1]
    # combine all other arguments into a single string
    combined_question = " ".join(sys.argv[2:])
    ask_ai(file_url, combined_question)


if __name__ == '__main__':
    main()

