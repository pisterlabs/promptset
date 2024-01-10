#%% pip installation
# !pip install -qU \
#   transformers==4.31.0 \
#   sentence-transformers==2.2.2 \
#   pinecone-client==2.2.2 \
#   datasets==2.14.0 \
#   accelerate==0.21.0 \
#   einops==0.6.1 \
#   langchain==0.0.240 \
#   xformers==0.0.20 \
#   bitsandbytes==0.41.0

#%% INIT HF Embedding Pipeline
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

print(f'device: {device}')


# %% Testing the embedding
docs = [
    "this is one document",
    "and another document"
]

embeddings = embed_model.embed_documents(docs)

print(f"We have {len(embeddings)} doc embeddings, each with "
      f"a dimensionality of {len(embeddings[0])}.")

# %% GET FAKE DATA

from datasets import load_dataset

data = load_dataset(
    'jamescalam/llama-2-arxiv-papers-chunked',
    split='train'
)

# %% SET UP FAISS VECTOR INDEX
# SKIP THIS IF YOU HAVE RUN IT ONCE

model_id = embed_model_id
embedding = embed_model

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

import numpy as np

data_pd = data.to_pandas()
texts = data_pd['chunk'].to_numpy()

# create vector database
def create_vector_database(texts, embeddings):
    # TODO splitter algorithm
    # loader = DirectoryLoader(DATA_PATH, glob = '*.pdf', loader_cls= PyPDFLoader)
    # documents = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter (chunk_size = 500, chunk_overlap = 50)
    # texts = text_splitter.split_documents(documents)

    db = FAISS.from_texts(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

create_vector_database(texts, embedding)



# %%

# !pip install chainlit
# !pip install ctransformers

# %%

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers # to use CPU only
from langchain.chains import RetrievalQA
import chainlit as cl
import transformers


DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """ Use the following pieces of information to answer user's question.
If you don't know the answer, just say you don't know. Don't make up information yourself.

Context: {context}
Question: {question}

Only returns helpful answer below and nothing else.
Helpful answer:
"""

model = "meta-llama/Llama-2-13b-chat-hf"
model_id = model

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate.from_template(template=custom_prompt_template)
    return prompt


def load_llm():
    model_id = 'meta-llama/Llama-2-13b-chat-hf'
    hf_auth = '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'

    model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    # quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
    )

    return model

def load_llm_ctra_backup():
    """
    loading CTransformers model
    """
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5

    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', # you can also change this to map reduce
        retriever = db.as_retriever(search_kwargs = {'k':3}),
        return_source_documents = True,        # retriever will ensure that llm will retrieve the information from the document that we have
        chain_type_kwargs = {'prompt' : prompt}
    )
    return qa_chain

#output function
def final_result(query):
    qa_result = retrieval_qa_chain()
    response = qa_result({'query': query})
    return response


# %% CALL LLM ################################################

final_result ("what is amino acid?")


# %%
