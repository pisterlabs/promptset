from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.llms import Ollama, OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma

from matbot.chains import APICallingChain

load_dotenv()

llms = {
    "openai": OpenAI(temperature=0),
    "mistral": Ollama(
        model="mistral", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    ),
}

persist_directory = "./docs/chroma"
embedding = OllamaEmbeddings(model="mistral")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

template = """You are a Retrieval-Augmented Generation chatbot that answers questions on
documents provided to you. Act as an expert in the subject matter of the document
discussed. If a question is not relevant for the document or if it cannot be answered
using the information of the document, please do not answer the question and politely provide
the reason.
Document: {context}

Question: {question}
Helpful Answer:
"""

prompt = PromptTemplate.from_template(template)

llm_name = "mistral"
llm_chain = RetrievalQA.from_chain_type(
    llm=llms[llm_name],
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "verbose": True},
)
api_url = "http://localhost:5000/chat"
app_chain = APICallingChain(api_url=api_url)
