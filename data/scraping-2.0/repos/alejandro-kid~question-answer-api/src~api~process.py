import fitz
import os
import openai
from openai import ChatCompletion
from flask import Response, current_app, json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.chains.question_answering import load_qa_chain
from config.settings import LLMS_MODEL_PATH, OPENAI_API_KEY

def process_files(document):
    try:
        if os.listdir(get_temp_folder()):
            os.remove(
                get_temp_folder() + "/" + os.listdir(get_temp_folder())[0]
            )

        temp_file = os.path.join(get_temp_folder(), document.filename)
        document.save(temp_file)

        data = {
                "success": True,
                "message": "Document processed successfully"
        }
        response = Response(json.dumps(data), 200, mimetype='application/json')
    except Exception as e:
        data = {
            "success": False,
            "error_message": str(e)
        }
        response = Response(json.dumps(data), 500, mimetype='application/json')
    return response

def query(query_text):
    try:

        docs = get_similarity_search(query_text)

        full_text = [ doc.page_content for doc in docs]

        data = {
                "success": True,
                "message": "Query processed successfully",
                "full_text": full_text
        }
        response = Response(json.dumps(data), 200, mimetype='application/json')
    except Exception as e:
        data = {
            "success": False,
            "error_message": str(e)
        }
        response = Response(json.dumps(data), 500, mimetype='application/json')
    return response


def query_plus(query_text):
    try:

        docs = get_similarity_search(query_text)

        llm = GPT4All(model=LLMS_MODEL_PATH, backend="gptj", verbose=False)
        chain = load_qa_chain(llm, chain_type="stuff")
        res = chain.run(input_documents=docs, question=query)

        data = {
                "success": True,
                "message": "Query processed successfully",
                "answer": res
        }
        response = Response(json.dumps(data), 200, mimetype='application/json')
    except Exception as e:
        data = {
            "success": False,
            "error_message": str(e)
        }
        response = Response(json.dumps(data), 500, mimetype='application/json')
    return response

def query_chatgpt(query_text):
    try:

        docs = get_similarity_search(query_text)
        full_text = [ doc.page_content for doc in docs]

        openai.api_key = OPENAI_API_KEY

        systemContent = ("You are a helpful assistant.You get your knowledge from the "
                         "following information delimited between three ticks.\n"
                         "```{}```\n The user will ask you questions about this"
                         " information and you should reply in a concise way. If you "
                         "can't deduce the answer using only this provided information,"
                         " just say you don't have the knowledge.".format(full_text))

        completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": systemContent},
            {"role": "user", "content": query_text}
        ]
        )

        data = {
                "success": True,
                "message": "Query processed successfully",
                "answer": completion.choices[0].message
        }
        response = Response(json.dumps(data), 200, mimetype='application/json')
    except Exception as e:
        data = {
            "success": False,
            "error_message": str(e)
        }
        response = Response(json.dumps(data), 500, mimetype='application/json')
    return response

def get_similarity_search(query_text) -> list[fitz.Document]:

    doc = fitz.Document()
    if os.listdir(get_temp_folder()):
        doc = fitz.Document(
            get_temp_folder() + "/" + os.listdir(get_temp_folder())[0]
        )
    else:
        raise Exception("No document found")

    embeddings = GPT4AllEmbeddings()
    n = doc.page_count

    doc_content = ""
    for i in range(0, n):
        page_n = doc.load_page(i)
        page_content = page_n.get_text("text")
        doc_content += page_content + "\n"

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(doc_content)

    document_search = FAISS.from_texts(texts, embeddings)

    docs = document_search.similarity_search(query_text)

    return docs

def get_temp_folder() -> str:
    dir = current_app.root_path + "/temp"
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    return dir

def delete_file(path:str) -> None:
    
    if os.path.exists(path):
        os.remove(path)
