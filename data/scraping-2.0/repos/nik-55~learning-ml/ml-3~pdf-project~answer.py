from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

repo_id = "google/flan-t5-xxl"

memory = ConversationBufferMemory(memory_key="chat_history")
embeddings = HuggingFaceEmbeddings()
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 1000})
prompt_template = PromptTemplate.from_template(
    "Give the answer to the question: {question} based on the following text: {content}"
)
llm_chain = LLMChain(prompt=prompt_template, llm=llm)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")


def getAnswer(question, file):
    db = FAISS.load_local("./store/"+file.name, embeddings)
    conversational_chain = ConversationalRetrievalChain(
        llm, memory=memory, retriever=db.as_retriever()
    )
    docs = db.similarity_search(question)
    content = " ".join([doc.page_content for doc in docs])
    return llm_chain.run({
        "question": question,
        "content": content
    })

def uploadedFile(file):
    reader = PdfReader(file)
    content = " ".join([page.extract_text() for page in reader.pages])
    docs = text_splitter.split_text(content)
    db = FAISS.from_texts(docs, embeddings)
    db.save_local("./store/"+file.name)