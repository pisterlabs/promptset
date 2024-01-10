# Import necessary modules
from dotenv import load_dotenv
import os

from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

load_dotenv('.env')

# Initialize language model
# llm = OpenAI(model_name="text-davinci-003", temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="cv.pdf")
document = document_loader.load()

# Load the summarization chain
# Summarize the document
summarize_chain = load_summarize_chain(llm)

with get_openai_callback() as cb:
	summary = summarize_chain(document)
	print(summary['output_text'])
	print(cb)