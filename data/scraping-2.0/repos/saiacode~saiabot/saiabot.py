import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#carga variables de ambiente, api key de openai: OPENAI_API_KEY
load_dotenv()

#lee el pdf y lo convierte a texto
doc_reader = PdfReader('./SAIA.pdf')

raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

#separa el texto en chunks de 1000 caracteres
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

#se utiliza FAISS como DB vectorial, podría usarse otras opciones tambien como chromadb, pinecone, etc
docsearch = FAISS.from_texts(texts, embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

prompt_template = """Usa las siguientes piezas de contexto para responder las preguntas. 
Utiliza un lenguaje coloquial y formal.
Sino conoces la respuesta, solamente di que no sabes la respuesta en este contexto.
Si preguntan por las redes sociales de SAIA, responder en bullets points y siempre colocar los links a las redes sociales.
{context}

Pregunta: {question}
Respuesta:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": prompt}

#creo instancia de RetrievalQA con el modelo de openai, el tipo de cadena, el retriever y los parametros del chain_type
rqa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, max_tokens=512), 
    chain_type="stuff", 
    retriever=retriever, 
    chain_type_kwargs=chain_type_kwargs
)

#creo la app de streamlit
st.title("Asistente SAIA")
st.write("Soy el BOT de SAIA. Preguntame lo que quieras!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_response(query):
    response = rqa(query)['result']
    return response


if prompt := st.chat_input("En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Escribiendo ..."):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = get_response(prompt)

            full_response += assistant_response

            message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})    
    