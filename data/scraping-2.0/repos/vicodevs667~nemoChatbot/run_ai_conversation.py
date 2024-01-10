from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import os
import sys
import gradio as gr
import time
import utils


os.environ["HUGGINGFACEHUB_API_TOKEN"] = utils.get_huggingface_api_key()
repo_id = "google/flan-t5-large"

loader = DirectoryLoader('data/',
                         glob="*.pdf",
                         loader_cls= PyPDFLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)

text_chunks = text_splitter.split_documents(documents)
#print(len(text_chunks))

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})

vector_store = FAISS.from_documents(text_chunks, embeddings)

#print(text_chunks)
query="lugares para hacer turismo en México"
docs = vector_store.similarity_search(query)

"""
llm = CTransformers(model = "model/pytorch_model-00001-of-00002.bin",
                    model_type="llama",
                    max_new_tokens=512,
                    temperature=0.2,
                    repetition_penalty=1.13
                    )
#llm = HuggingFacePipeline.from_model_id(model_id="OpenAssistant/stablelm-7b-sft-v7-epoch-3", task="text-generation", model_kwargs={"temperature": 0.2, "max_length": 2048, 'device_map': 'auto'})
"""

llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs= {"temperature":0.2, "max_length": 512})

template="""Eres un asistente de turismo a México y debes responder la pregunta tomando en cuenta el contexto dado.
Si la pregunta es hola debes responder saludando al usuario y pregúntandole en que le puedes ayudar.
Si la pregunta no puede ser respondida usando la información proporcionada, responde con "chale, ni idea wey"

Contexto:{context}
Pregunta:{question}

Respuesta (escribe como mexicano informal):
"""

qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k': 8}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': qa_prompt})

"""while True:
    user_input=input(f"prompt:")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = chain({'query':user_input})
    print(f"Answer:{result['result']}")
"""
def bot(query):
    llm_response = chain({'query':query})
    return llm_response['result']

with gr.Blocks(title='Nemo Chatbot de Turismo') as demo:
    gr.Markdown("# Chatbot - México a tu alcance")
    chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
    msg = gr.Textbox(label="Usuario", placeholder="Ingrese su consulta")
    clear = gr.ClearButton([msg, chatbot], value="Limpiar contenido")

    def respond(message, chat_history):
        bot_message = bot(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()