import os
import together
import shutil
import logging
import time
import textwrap
from typing import Any, Dict, List, Mapping, Optional
from fastapi import FastAPI

from pydantic import Extra, Field, root_validator, model_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv

# set your API key
load_dotenv()
os.environ['TOGETHER_API_KEY']
together.api_key = os.environ["TOGETHER_API_KEY"]


class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

#     @model_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text

################################################################################
# split documents into chunks, create embeddings, store embeddings in chromaDB #
################################################################################

chunk_size = 1000
chunk_overlap=200
loader = DirectoryLoader('/home/austin/code/ai/RAGS/llm/stage_data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f' number of documents {len(documents)}')
#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)
print(f' number of chunks {len(texts)}')


# Instantiate embeddings model
model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

# create db
# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db'

# persist_directory = '/home/austin/code/ai/RAGS/db'
## Here is the nmew embeddings being used
embedding = model_norm

t1 = time.perf_counter()
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)



t2 = time.perf_counter()
print(f'time taken to embed {len(texts)} chunks:',t2-t1)
print(f'time taken to embed {len(texts)} chunks:,{(t2-t1)/60} minutes')
print(f'time taken to embed {len(texts)} chunks:,{((t2-t1)/60)/60} hours')


##############################################################
# move pdf files from staging directory to archive directory #
##############################################################
def list_files(directory):
    """Return a list of filenames in the given directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

src_dir = '/home/austin/code/ai/RAGS/llm/stage_data'
dst_dir = '/home/austin/code/ai/RAGS/llm/data'

# List files in both directories before moving

# Check if the destination directory exists, if not create it
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# List all files in the source directory
files = list_files(src_dir)

# Move each file to the destination directory
for file in files:
    src_file_path = os.path.join(src_dir, file)
    dst_file_path = os.path.join(dst_dir, file)
    shutil.move(src_file_path, dst_file_path)


print(f"Moved {len(files)} files from {src_dir} to {dst_dir}.")
print(f"Files moved: {files}")
print("\n".join(list_files(src_dir)))

# retriever = vectordb.as_retriever(search_kwargs={"k": 5})
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """You are both a professor of medicine and a highly esteemed researcher in human genetic engineering. Your goal is to invent novel treatments for human cancers.

Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, provide what information is needed for the question to be answered. If you don't know the answer to a question, please don't share false information.

Your superior logic and reasoning abilities coupled with you vast knowledge in biology, genetics, and medicine allow you to conduct innovative experiments resulting in significant advancements in medicine.
"""

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

get_prompt(instruction, DEFAULT_SYSTEM_PROMPT)


from langchain.prompts import PromptTemplate
from langchain.schema import prompt

llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature = 0.1,
    max_tokens = 2024
)


prompt_template = get_prompt(instruction, DEFAULT_SYSTEM_PROMPT)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       chain_type_kwargs=chain_type_kwargs,
                                       return_source_documents=True)

## Cite sources
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


query = "provide a mock case study for a bladder cancer patient who also has CKD, and provide a personalized treatment plan with timelines and dosages?"
llm_response = qa_chain(query)
process_llm_response(llm_response)