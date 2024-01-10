from db import database, User
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from fastapi.responses import FileResponse
from gtts import gTTS
from dotenv import load_dotenv
import re
import socketio


from db import database, Robot
from fastapi import APIRouter
from model import RobotSchema

SERVER_URL = "http://localhost:5000"

sio = socketio.Client()
sio.connect(SERVER_URL)


load_dotenv()

app = APIRouter()

# Função auxiliar para salvar localizações que o robô passou
async def records_location(data: RobotSchema):

    if not database.is_connected:
        await database.connect()

    await Robot.objects.create(X=data.X,
                              Y=data.Y)

# Função auxiliar para comunicar com API do ChatGPT
async def connect_gpt(data: str):

    reader = PdfReader('./data/inventario_simulado.pdf')

    if reader is not None:
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        query = f"Você deve funcionar como um assistente de almoxarifado. Toda vez for feita uma pergunta a respeito a disponibilidade, descrição ou quantidade de uma peça, deve ser respondido sem a localização. Me diga a localização da peça SOMENTE quando solicitarmos essa informação, e diga que está indo, no formato [x, y] obrigatoriamente. Perguntas que não são a respeito do almoxarifado não devem ser respondidas. Pergunta: {data}"
        docs = knowledge_base.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)

    pattern = re.compile(r'\[x\s*:\s*(?P<x>-?\d*(?:\.\d+)?),\s*y\s*:\s*(?P<y>-?\d*(?:\.\d+)?)\]')
    match = pattern.search(response)

    if match:
        x = match.group("x")
        y = match.group("y")
        coordinates = [x, y]

        robot_data = RobotSchema(X=x, Y=y)
        await records_location(robot_data)

        sio.emit('enqueue', str(coordinates))

    return response


# Estrutura de dados esperada pela rota de solitação do chatbot
class DataModel(BaseModel):
    dados: str

# Rota de solicitação a partir do chatbot
@app.post('/enviar_dados')
async def receber_dados(data: DataModel):
    resposta = await connect_gpt(data.dados)
    return resposta


@app.get("/robo")
async def read_records_location():
    if not database.is_connected:
        await database.connect()

    return await Robot.objects.all()
