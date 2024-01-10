import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.prompts import PromptTemplate
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index import StorageContext, load_index_from_storage
from glob import glob
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.text_splitter import TokenTextSplitter, SentenceSplitter
import torch
import re


PROMT = "Ответь на вопрос: {query_str}"

SP = "Ты умный ассистент которого зовут Резюмируй. Ты любишь давать рекомендации по поводу улучшения резюме опираясь на существующие."

MODEL_NAME = "IlyaGusev/saiga2_7b_lora"


def response(question):
    documents = SimpleDirectoryReader("data").load_data()
    tok = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru", use_fast=False)
    parser = SimpleNodeParser.from_defaults(
        separator="\n", chunk_size=256, chunk_overlap=32
    )
    # metadata_extractor=metadata_extractor)
    nodes = parser.get_nodes_from_documents(documents, show_proggres=True)
    len(nodes)

    for node in nodes:
        node.text = re.sub("\\n", " ", node.text)
        node.text = re.sub(" +", " ", node.text)

    config = PeftConfig.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map="auto",
    )
    llm = PeftModel.from_pretrained(llm, MODEL_NAME, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    llm = HuggingFaceLLM(
        model=llm,
        tokenizer=tokenizer,
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.3, "do_sample": True},
        query_wrapper_prompt=PROMT,
        system_prompt=SP,
        device_map="auto",
        tokenizer_kwargs={"max_length": 4096},
        model_kwargs={"torch_dtype": torch.float16},
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=256, llm=llm, embed_model="local:cointegrated/LaBSE-en-ru"
    )
    torch.cuda.empty_cache()
    index = VectorStoreIndex(nodes=nodes, service_context=service_context)
    query_engine = index.as_query_engine(similarity_top_k=5)
    res = query_engine.query(question)
    return res.response
