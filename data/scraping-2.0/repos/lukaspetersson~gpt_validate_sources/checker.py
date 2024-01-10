from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ChatVectorDBChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from dotenv import dotenv_values
import os

env_var = dotenv_values(".env")
openai_token = env_var["OPENAI_TOKEN"]
os.environ["OPENAI_API_KEY"] = openai_token

##
pdf_path = "./input.pdf"
loader = PyPDFLoader(pdf_path)
tokens_per_page = 500
text_splitter = TokenTextSplitter(chunk_size=tokens_per_page, chunk_overlap=0)
pages = loader.load_and_split(text_splitter)

##
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory="vector_db")
vectordb.persist()

pdf_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0.9, model_name="gpt-4"), vectordb, return_source_documents=True)

##
claim = "Various studies have explored the impact of robot gaze on HRI, finding that gaze direction can influence perceived personality, trustworthiness, and engagement"
query = "Which sentence in the context best supports the following claim: " + claim
result = pdf_qa({"question": query, "chat_history": ""})
print("Answer:")
print(result["answer"])
