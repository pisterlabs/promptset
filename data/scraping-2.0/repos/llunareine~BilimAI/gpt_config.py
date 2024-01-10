import json
import os
import nest_asyncio
from langchain.document_loaders import WebBaseLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from links import get_link
import sqlite3 as sq
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def nfacotiral(question):
    loader = WebBaseLoader(
        ["https://www.nfactorial.school/"]
    )
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(docs, embeddings)
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    response = qa(question)
    return response["result"]


def chat_query(question: str):
    if ("courses" in question.lower() or "course" in question.lower()) and "nfactorial" not in question.lower():
        try:
            sqliteConnection = sq.connect('db.sqlite3')
            cursor = sqliteConnection.cursor()
            cursor.execute("SELECT name, description, price FROM api_course")
            datas = cursor.fetchall()
            if datas:
                json_loader_list = []
                for data in datas:
                    json_loader_list.append({'courses': [{
                        "name_course": data[0],
                        "course_description": data[1],
                        "course_price": data[2]
                    }]})
                json_loader = json.dumps(json_loader_list)
                with open("courses/courses.json", "w") as f:
                    f.write(json_loader.replace("[", "").replace("]", ""))
                loader = JSONLoader(
                    file_path='courses/courses.json',
                    jq_schema='.',
                    text_content=False)
                docs = loader.load()
                embeddings = OpenAIEmbeddings()
                docsearch = Chroma.from_documents(docs, embeddings)
                retriever = docsearch.as_retriever(search_kwargs={"k": 3})

                llm = ChatOpenAI()
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                response = qa(question)
                return response["result"]
            else:
                return nfacotiral(question)
        except sq.DatabaseError as e:
            pass
    elif ("courses" in question.lower() or "course" in question.lower()) and "nfactorial" in question.lower():
        return nfacotiral(question)
    loader = WebBaseLoader(
        get_link(question)
    )

    data = loader.aload()
    text_spliter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_spliter.split_documents(data)
    prompt = """Prompt:you are an expert on the issues of unified national testing and admission to universities in 
    Kazakhstan. Please do not hesitate to ask any questions related to these topics, university admission procedures 
    or recommendations for choosing a university or specialty, as well as preparing for the UNT. if the questions are 
    not about these topics, just answer I'm sorry, I do not know the answer to your question.
        
    User:{question}
    """

    prompt_template = ChatPromptTemplate.from_template(prompt)

    question = prompt_template.format_messages(question=question)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(docs, embeddings)
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    response = qa(question[0].content)
    return response["result"]
