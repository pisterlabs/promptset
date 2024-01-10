from langchain import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
import openai
import os

from langchain.vectorstores.base import VectorStoreRetriever

# read local .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Leemos el documento
template = '''
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use 150 word maximum to answer.
Always include the page where the answer is.
Always say "thanks for asking!" at the end of the answer.
Context is delimited by triple dollar signs.

$$${context}$$$

Question: {question}
Helpful Answer:
'''

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

loader = PyPDFLoader('./pdfs/napoleon.pdf')
data = loader.load()

pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

all_splits = pdf_splitter.split_documents(data)

# Añadimos memoria.
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

persist_directory = './pdfs/chroma/'

vectorstore = Chroma.from_documents(documents=data,
                                    embedding=OpenAIEmbeddings(),
                                    persist_directory=persist_directory)

vectorstore2 = FAISS.from_documents(documents=data,
                                    embedding=OpenAIEmbeddings()
                                    )



question = '¿Podrías resumirme el capiturlo Las aventuras española y rusa, y decirme en que pagina empieza y en que página acaba?'

docs = vectorstore2.as_retriever(search_type='similarity', search_kwargs={'k': 4}, include_metadata=True)

# Mostramos los metadatos de los chucnks selecionados.
for doc in docs:
    print(doc)

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, verbose=True)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       retriever=docs,
                                       chain_type='stuff',
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
                                       verbose=True
                                       )

result = qa_chain({'query': question, 'input_documents': docs, 'verbose': True})
print(result)
print(result['result'])
