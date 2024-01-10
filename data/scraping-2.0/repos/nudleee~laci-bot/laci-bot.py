import os 
import openai
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import  ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
import gradio as gr


openai.api_key  = os.environ['OPENAI_API_KEY']
llm_name = 'gpt-3.5-turbo'
persist_directory = 'files/chroma'
embedding = OpenAIEmbeddings()

def load_db():
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory,embedding_function=embedding)
    else:
        loader_kwargs={"encoding": "utf_8"}
        pdf_loader = PyPDFLoader('files/Szakmai_gyak_BSc_szabályzat_2014 után.pdf')
        pdf_docs = pdf_loader.load()
        csv_loader = DirectoryLoader('files/', glob="*.csv", loader_cls=CSVLoader, loader_kwargs=loader_kwargs)
        csv_docs = csv_loader.load()
        txt_loader = DirectoryLoader('files/', glob='*.txt', loader_cls=TextLoader, loader_kwargs=loader_kwargs)
        txt_docs=txt_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100,
            length_function=len,
            is_separator_regex=True,
            separators=["\n\s*\n", "\n\s*", "\n", "\n\n\n\n", "\n\n\t\t"]
        )
        pdf_splits = text_splitter.split_documents(pdf_docs)
        txt_splits = text_splitter.split_documents(txt_docs)
        data = []
        data.extend(pdf_splits)
        data.extend(txt_splits)
        data.extend(csv_docs)
        vector_db = Chroma.from_documents(documents=data, embedding=embedding, persist_directory=persist_directory)
        vector_db.persist()
        return vector_db

vector_db = load_db()
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
llm = ChatOpenAI(model=llm_name, temperature=0.3, request_timeout=120)

template = """A BME VIK szakmai gyakorlattal kapcsolatos kérdéseket megválaszoló chatbot vagy. A feladatod, hogy a kérdést alakítsd át az előzmények alapján, hogy értelmes legyen. 
Ha nem tudsz a szakmai gyakorlatra vonatkozó kérdést előállítani, akkor legyen az új kérdés: "Nem tudok válaszolni".

Előzmények: {chat_history}

Kérdés: {question}

Új kérdés:"""

QG_CHAIN_PROMPT = PromptTemplate(input_variables=['chat_history', 'question'],template=template)
question_generator_chain = LLMChain(llm=llm, prompt=QG_CHAIN_PROMPT)

qa_template = """A BME VIK szakmai gyakorlattal kapcsolatos kérdéseket megválaszoló chatbot vagy és a neved Laci-bot. Válaszolj a kérdésre magyarul, amihez az alábbiakban találasz releváns információkat,
de ha nem tudsz válaszolni, akkor ne próbálj meg kitalálni valamit, hanem mondjad "Sajnos nem tudok ezzel kapcsolatban információval szolgálni".

Dokumentumok: {documents}

Előzmények: {chat_history}

Kérdés: {question}

Válasz:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=['context', 'chat_history', 'documents'],template=qa_template)
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='documents',)

chain = ConversationalRetrievalChain(
    combine_docs_chain=doc_chain,
    question_generator=question_generator_chain,
    retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"fetch_k":8, "k": 5}),
    memory=memory,
    return_generated_question=True,
    return_source_documents=True,
)

def predict(message, chat_history):
    result = chain({'question': message, 'chat_history': chat_history})
    answer = result['answer']
    chat_history.append((message, answer))
    return '', chat_history

with gr.Blocks() as chat_interface:
    chatbot = gr.Chatbot(label='Laci-bot')
    msg = gr.Textbox(label='Kérdés', placeholder="Kérdezd Lacit a BME VIK szakmai gyakorlatával kapcsolatban")
    clear = gr.ClearButton(value='Törlés', components=[msg,chatbot])            
    msg.submit(predict, [msg, chatbot], [msg, chatbot])
   
chat_interface.launch()