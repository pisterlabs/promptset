"""
https://python.langchain.com/en/latest/getting_started/getting_started.html

pip install langchain
pip install openai


# طريقة عمل بيئة البايثون الخاصة بالمشروع
# How to make a virtual python env for your project.

> python -m venv venv

>
On Windows: venv\Scripts\activate.bat
On Unix or Linux: source venv/bin/activate

> pip install mylibrary

# ممكن إخراج هذه المكتبات فى ملف
# export lib name as requirements file
> pip freeze > requirements.txt

# Stop virtual env
deactivate
"""
from dotenv import load_dotenv
import os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model_name='gpt-4', temperature=0.9)

with open('/Users/khaledlela/Downloads/test.txt') as file:
    source = file.read()

text_splitter = CharacterTextSplitter(separator=" ", chunk_size=12000)
# Split text into smaller chunks and create Document objects
texts = text_splitter.split_text(source)
docs = [Document(page_content=t) for t in texts]

model = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
model.run(docs)

print(len(docs))

for doc in docs:
    print(doc)
prompt = "What the main language for GPT?"
print(llm(prompt))
# The main language used by GPT (Generative Pre-trained Transformer) is Python.
