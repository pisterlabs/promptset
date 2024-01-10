import openai
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(temperature=0.0, model_name='gpt-4')
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 20,
)

map_template_string = """
Sei un esperto legale nel sistema legale italiano. Riassumi in termini semplici che evidenzino i punti chiave
del seguente documento legale delimitato da triple backticks. Nota che il documento è diviso in più parti,
riceverai un set di documenti.
```${documents}```
"""

reduce_template_string = """"
Il seguente (fra triple backticks) è un insieme di riassunti di parti di un documento legale italiano.
I riassunti sono scritti allo scopo di semplificare la comprensione del documento.
Prendi questi documenti come input e scrivi un documento finale riassuntivo.
```${documents}```
"""

map_prompt_template = ChatPromptTemplate.from_template(map_template_string)
reduce_prompt_template = ChatPromptTemplate.from_template(reduce_template_string)
map_chain = LLMChain(llm=llm, prompt=map_prompt_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt_template)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="documents"
)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=4000,
)
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="documents",
    return_intermediate_steps=False,
)

def simplify(parsed_document):
    splitted_document = splitter.create_documents([parsed_document])
    splits = map_reduce_chain.run(splitted_document)
    print(splits)
    return "lol"
