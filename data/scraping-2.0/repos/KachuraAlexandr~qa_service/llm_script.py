import os
import pandas as pd
import re
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.output_parsers import PydanticOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uvicorn


def load_db(docs_dir: str) -> Chroma:
    cards = pd.read_csv(os.path.join(docs_dir, 'cards.csv'))
    cards = [
        Document(page_content=f'{service} {condition} {tariff}')
        for service, condition, tariff in
        zip(
            cards['Service'].values,
            cards['Condition'].values,
            cards['Tariff'].values
        )
    ]

    doc1 = PyPDFLoader(os.path.join(docs_dir, 'doc1.pdf')).load()[5:]
    doc1 = [page.page_content.replace('\xa0', ' ') for page in doc1]
    doc1 = [re.sub(r'[0-9]+ из [0-9]+', '', page) for page in doc1]
    doc1 = sum([re.split(r'\n[0-9]+.[0-9]+. ', page) for page in doc1], [])
    doc1 = sum([re.split(r'\n[0-9]+. ', s) for s in doc1], [])
    doc1 = [Document(page_content=re.sub('\n', ' ', s)) for s in doc1]

    doc2 = PyPDFLoader(os.path.join(docs_dir, 'doc2.pdf')).load()
    doc2 = [page.page_content.replace('\xa0', ' ') for page in doc2]
    doc2 = [re.sub('[0-9]+ из [0-9]+', '', page) for page in doc2]
    doc2 = sum([re.split(r'\n[0-9]+. ', s) for s in doc2], [])
    doc2 = [Document(page_content=re.sub('\n', ' ', s)) for s in doc2]

    embeddings_model = HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'})
    return Chroma.from_documents(cards + doc1 + doc2, embeddings_model)


def get_tariffs_info(db: Chroma, query: str):
    results = db.similarity_search(query)
    results = ' '.join([res.page_content for res in results])
    return results


class Message(BaseModel):
    user_id: str
    message: str


db = load_db('./data')

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path='./weights/llama-2-7b-chat.Q4_K_M.gguf',
    temperature=0.01,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048
)

app = FastAPI()
redis_backed_dict = {}
parser = PydanticOutputParser(pydantic_object=Message)
prompt = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template(
            'Ты - оператор в банке Тинькофф. Ты отвечаешь пользователям на вопросы '
            'об условиях обслуживания клиентов и тарифах на услуги банка.'
            'Отправь информацию в формате {format_instructions} в чат, '
            'если пользователь хочет получить информацию об условиях обслуживания '
            'или тарифе на услугу '
            'и у тебя есть вся необходимая информация. '
            'Отвечай только на русском языке. '
            'Для ответа используй только информацию, которая предоставлена ниже. '
            '{context}'
        ,),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{question}')
    ],
    partial_variables={
        'format_instructions': parser.get_format_instructions()
    }
)

@app.post('/message')
def message(message: Message):
    user_id = message.user_id
    message = message.message

    memory = redis_backed_dict.get(
        user_id,
        ConversationBufferMemory(
            memory_key='chat_history',
            input_key='question',
            return_messages=True
        )
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    docs = db.similarity_search(message)
    docs_page_content = " ".join([d.page_content for d in docs])
    ai_message = conversation({'question': message, 'context': docs_page_content})

    try:
        command: Optional[Message] = parser.parse(ai_message['text'])
        if command is not None:
            return {'message': 'Условия по интересующей вас услуге предоставлены.'}
    except:
        pass

    return {'message': ai_message['text']}
