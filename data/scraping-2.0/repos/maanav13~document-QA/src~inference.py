from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnablePassthrough

# Load PDF and split into page chunks.
loader = PyPDFLoader("<file path>")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)
pages = loader.load_and_split(text_splitter=text_splitter)

# Initialize model to create embeddings of chunks.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create index of embeddings
db = FAISS.from_documents(pages, embeddings)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Initialize the LLM
ollm = Ollama(model='llama2-7b-gguf', base_url='http://localhost:11434', callback_manager=callback_manager)


# Provide the following system instructions to the LLM
template = """<<SYS>>
You are a helpful, respectful and honest assistant.
You will recieve questions in the following format:
"Context: <context>
Question: <question>"
Use the provided context to answer the question.
You will answer in the following format: 
"Answer: <answer>" 
where <answer> is the answer to <question>.
Ensure your answers are three sentences maximum and keep the answer as concise as possible.
<</SYS>>
[INST] 
Context: {context}
Question: {question}
[/INST]
"""

prompt = PromptTemplate.from_template(template)
chain_type_kwargs = {"prompt": prompt}


question = """
<question>
"""

retriever = db.as_retriever()
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | ollm 
)

rag_chain.invoke(question)
