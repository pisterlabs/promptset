import gradio as gr
from openai import OpenAI
import docx2txt
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
import os
from langchain.embeddings import OpenAIEmbeddings


api_key = "sk-"  # Replace with your key
db = None
os.environ["OPENAI_API_KEY"] = api_key

def read_text_from_file(file_path):
    # print(file_path)
    # Check the file type and read accordingly
    if file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)

    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
                
    return loader.load()

def load_documents(file_path):
    global db
    documents = []

    for files in file_path:
        file_contents = read_text_from_file(files)
        documents.extend(file_contents)

    # Split text from PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
    texts = text_splitter.split_documents(documents) 

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    print('file loading done')


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    filename = gr.File(file_count='multiple')

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(msg, history,filename):
        history_openai_format = []
        print('msg',msg)
        print('history', history[-1][0])

        query_results = db.similarity_search(history[-1][0])
        print('queryresults',query_results)

        for query_result in query_results:
            history_openai_format.append({"role": "user", "content": query_result.page_content })
        
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human })
            if assistant is not None:
                history_openai_format.append({"role": "assistant", "content":assistant})
               
        client = OpenAI(
        api_key=api_key,)
        response = client.chat.completions.create(
        messages=history_openai_format,
        model="gpt-3.5-turbo", # gpt 3.5 turbo
        # model="gpt-4",
        # model = "gpt-4-1106-preview", #gpt-4 turbo
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
    filename.change(load_documents,filename,[])
    
demo.queue()
demo.launch()
