import boto3
import urllib.parse
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
# from langchain.prompts.chat import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
    
# )
from langchain.schema import HumanMessage, SystemMessage

AWS_BUCKET = 'chatbot-data-storage'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')




def s3PdfDoader(s3_path):
    bucket = boto3.resource('s3').Bucket(AWS_BUCKET)
    pdf_obj = bucket.Object(s3_path).get()['Body'].read()
    tmp_path = f'/tmp/{s3_path.split("/")[-1]}'

    print('pdf_obj: ', pdf_obj)
    print('tmp_path: ', tmp_path)

    with open(tmp_path, 'wb') as f:
        f.write(pdf_obj)

    return PyPDFLoader(tmp_path)

def get_documents(text):
    chat = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

    # system = "入力された内容を次のフォーマットで内容ごとに日本語でまとめてグループ分けしてください。"
    # human = "{text}"

    # system_prompt = SystemMessagePromptTemplate.from_template(system)
    # temlate_prompt = SystemMessagePromptTemplate.from_template(template)
    # human_prompt = HumanMessagePromptTemplate.from_template(human)

    # chat_prompt = ChatPromptTemplate.from_messages([system_prompt, temlate_prompt, human_prompt])
    # res = chat(chat_prompt.format_prompt(text=text).to_messages())

    # print('AIMessage: ', res)



    message = [
        SystemMessage(
            content="以下にかかれている内容を箇条書きでまとめてください"
        ),
        # SystemMessage(
        #     content=template
        # ),
        HumanMessage(content=text)
    ]

    res = chat.invoke(message)

    print(res)
    return res



def handler(event, context):
    # embeddings = OpenAIEmbeddings()


    s3_path = urllib.parse.unquote(
        event['Records'][0]['s3']['object']['key']
    )

    print('s3_path: ', s3_path)

    
    loader = s3PdfDoader(s3_path)

    documents = loader.load()

    text = ''
    for x in documents:
        text = text + x.page_content


    get_documents(text)

    # Document(page_content='', metadata={'source': '', 'page': 0}),


    # vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    # print('vector_store: ', vector_store)

    # serialized_vector_store = vector_store.serialize_to_bytes()
    # print('serialized_vector_store: ', serialized_vector_store)
