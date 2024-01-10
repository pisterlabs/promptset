{%- if cookiecutter.secrets_store == "environment" %}
from dotenv import load_dotenv
{%- endif %}
{%- if cookiecutter.llm_provider == "Azure OpenAI" %}
from langchain.llms import AzureOpenAI
{%- endif %}
{%- if cookiecutter.use_vectorstore == 'FAISS' %}
from langchain.vectorstores import FAISS
{%- elif cookiecutter.use_vectorstore == 'Pinecone' %}
from langchain.vectorstores import Pinecone
{%- endif %}
{%- if cookiecutter.use_vectorstore == 'Cohere' %}
from langchain.embeddings import CohereEmbeddings
{%- endif %}
{%- if cookiecutter.use_memory == 'yes' %}
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
{%- endif %}
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter