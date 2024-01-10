from langchain.llms import LlamaCpp, OpenAI
from langchain.chat_models import ChatOpenAI # preferred over non chat
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings
from configparser import ConfigParser
import os

parser = ConfigParser()
conf_file = parser.read("config.ini")
if not conf_file:
    raise FileNotFoundError("config.ini file not found")


if parser['SETTINGS']['LLM'] == 'Llama':
    if int(parser['LLAMA']['gpu']):
        model_path = parser['LLAMA']['location']
        n_gpu_layers = int(parser['LLAMA']['n_gpu_layers'])
        n_batch = int(parser['LLAMA']['n_batch'])

        embeddings = LlamaCppEmbeddings(
                                        model_path=model_path,
                                        n_gpu_layers=n_gpu_layers,
                                        )

        llm = LlamaCpp(
                        model_path=model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_batch=n_batch,
                        verbose=True,
                        )
        
elif parser['SETTINGS']['LLM'] == 'OpenAI':
    os.environ["OPENAI_API_KEY"] = parser['OPENAI']['key']
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0)

