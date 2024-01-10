from dotenv import load_dotenv
import loggingService

from genai.extensions.langchain import LangChainInterface
from genai.model import Credentials, Model
from genai.schemas import GenerateParams

from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Weaviate

import os
from sentence_transformers import SentenceTransformer
import weaviate

load_dotenv()
logger = loggingService.get_logger()

api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", 'https://workbench-api.res.ibm.com')
class_name ='Livros'
memory = ConversationBufferMemory(memory_key="chat_history",input_key="question", output_key='answer', return_messages=True)
model_name = os.getenv('MODEL_NAME', 'bigscience/mt0-xxl')
weaviate_url = os.getenv("WEAVIATE_URL", 'http://127.0.0.1:8080')

client = weaviate.Client(url=weaviate_url,)

creds = Credentials(api_key, api_endpoint=api_endpoint)
params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=100,
    min_new_tokens=1,
    stream=False,
    temperature=0.7,
    top_k=50,
    top_p=1,
).dict()  # Langchain uses dictionaries to pass kwargs
print(api_endpoint)
print(model_name)
llm = LangChainInterface(model=model_name, credentials=creds, params=params)
vector_store = Weaviate(client=client, index_name='Livros', text_key='content')


pt1 = """Responda a pergunta a seguir de forma sucinta usando o contexto e histórico fornecidos. Caso não tenha certeza da resposta sinceramente diga que não possui informações suficientes sobre esse tema.

Contexto: {context}

Histórico: {chat_history}

Pergunta: {question}
Resposta:"""
# Histórico: {chat_history} "chat_history"
prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=pt1,
)

retriver = vector_store.as_retriever(search_kwargs={'score_threshold': 0.6})
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriver, memory=memory, combine_docs_chain_kwargs={"prompt": prompt}, verbose=True)

def get_llm_response(query: str, history) -> str:
    resposta = qa.run(query)
    
    return resposta

if __name__ == '__main__':
    print(get_llm_response('por que a casa do arthur dent ia ser demolida?'))
    # print(get_llm_response('por que arthur dent foi despejado?'))
