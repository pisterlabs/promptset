from typing import List
from langchain.llms import OpenAI

import tiktoken
from langchain.llms import llamacpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY", '')
if not api_key :
    load_dotenv(find_dotenv('.env.summarizer'))


llm = ChatOpenAI(temperature = 0, openai_api_key = api_key)

# local llm
# model_path = '/summarizer/local_llm/mistral-7b-openorca.Q4_0.gguf' #all-MiniLM-L6-v2-f16.gguf'

# n_gpu_layers = 40
# n_batch = 512
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = llamacpp.LlamaCpp(
#     model_path = model_path,
#     n_gpu_layers= n_gpu_layers,
#     n_batch = n_batch,
#     callback_manager = callback_manager,
#     verbose = True
# )

# Map prompt
map_template = """The follwing is a set of documents
{docs}
Based on this list of docs, please identify the main themes
Helpful Answer:"""

map_prompt = PromptTemplate.from_template(map_template)

# Reduce prompt
reduce_template = """The following is set of summaries:
{doc_summaries}
Take these summaries and distill it into a final, consolidated summary of the main themes.
The final answer is a single paragraph of about 100 words and must be in korean.
Helpful Answer:"""

reduce_prompt = PromptTemplate.from_template(reduce_template)

reduce_chain = LLMChain(llm = llm, prompt = reduce_prompt)

combine_documents_chain = StuffDocumentsChain(
    llm_chain = reduce_chain, document_variable_name = "doc_summaries"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain = combine_documents_chain,
    # if documents exceed context for 'StuffDocumentsChain'
    collapse_documents_chain = combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max = 256,
)

# 2. Map chain
map_chain = LLMChain(llm=llm, prompt = map_prompt)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain = map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The Variable name in the llm_chain to put the documents in
    document_variable_name = 'docs',
    # Return the results of the map steps in the output
    return_intermediate_steps = False,
)

tokenizer = tiktoken.get_encoding('cl100k_base')

def token_size(text) :
    tokens = tokenizer.encode(text)
    return len(tokens)

def split_docs(context : str) -> List[Document] :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 128,
        chunk_overlap = 25, 
        length_function = token_size
    )

    docs = [Document(page_content = x) for x in text_splitter.split_text(context)]

    return docs

## text split
def split_text(context : str) -> List[Document] :
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 128,
        chunk_overlap = 25, 
        length_function = token_size
    )

    docs = [Document(page_content = x) for x in text_splitter.split_text(context)]

    split_docs = text_splitter.split_documents(docs)

    return split_docs


def mapreduce_transcript(context : str) -> str :
    
    split_docs = split_text(context)
    sum_result = map_reduce_chain.run(split_docs)

    return sum_result


