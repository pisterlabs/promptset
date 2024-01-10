import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoConfig,
    pipeline,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import torch

torch.cuda.empty_cache()

_ = load_dotenv(find_dotenv())  # read local .env file


# ----------------------------
# DOES NOT WORK !!!
#
# ==============================WARNING: DEPRECATED!==============================
# WARNING! This version of bitsandbytes is deprecated. Please switch to `pip install bitsandbytes` and the new repo: https://github.com/TimDettmers/bitsandbytes
# ==============================WARNING: DEPRECATED!==============================
# An exception occurred
# module 'bitsandbytes.nn' has no attribute 'Linear8bitLt'
# ----------------------------


start = time.time()


# ----------------------------
# DATA
# ----------------------------

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = "db"
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
    anonymized_telemetry=False,
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    client_settings=chroma_settings,
)
retriever = db.as_retriever()


print("DB")

# ----------------------------
# LLM
# ----------------------------

# mosaicml/mpt-7b-chat > 13 GB ---------------------
# max block = 10 GB
# partial processing possible: https://huggingface.co/blog/accelerate-large-models


# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )


try:
    model_id = "mosaicml/mpt-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    print("tokenizer")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.init_device = "cuda"
    print("config")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        # quantization_config=quantization_config,
        load_in_8bit=True,
        max_memory=f"{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB",
    )
    # .to("cuda")
    print("model")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=100,
        device_map="auto",
        # device="cuda",
    )
    print("pipeline")
    local_llm = HuggingFacePipeline(pipeline=pipe)
except Exception as e:
    print("An exception occurred")
    print(e)


print("LLM")

# ----------------------------
# CHAIN = Option 1. - ConversationalRetrievalChain
# ----------------------------

qa = ConversationalRetrievalChain.from_llm(
    llm=local_llm, retriever=retriever, chain_type="stuff"
)

print("CHAIN")

questions = [
    "What is this app about?",
    "What Assets are used in the app?",
    "Is in the code use async/await?",
    "What is the most efficient class in the code?",
    "What is the class hierarchy?",
    "What classes are derived from the DBObject class?",
]
chat_history = []

for question in questions:
    print(f"Question: {question} \n")
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"Answer: {result['answer']} \n")
    print(" ")


end = time.time()
print(f"NN takes: {end - start} sec.")
