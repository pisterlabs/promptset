import projsecrets
import json
from pathlib import Path
import pprint
import pdb
from typing import Any
from etl import markdown, pdfs, shared, videos
import textwrap
import docstore
import time
import vecstore
from utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, TextStreamer

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
import langchain

embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          token=True)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16,
                                             token=True,
                                             #  load_in_8bit=True,
                                             #  load_in_4bit=True,
                                             )
streamer = TextStreamer(tokenizer, skip_prompt=True)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=4096,
                do_sample=True,
                # temperature=0.1,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                )

llm = HuggingFacePipeline(pipeline=pipe)

from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

pretty_log("connecting to vector storage")
vector_index = vecstore.connect_to_vector_index(vecstore.INDEX_NAME, embedding_engine)
pretty_log("connected to vector storage")
pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

###############




########## THE LAMA 2 DEMO ############## - MORE GENERIC AND CUSTOMIZED
langchain.debug=False

llama_docs_template = """
[INST]Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
{context}
Question: {question} [/INST]
"""
llama_docs_prompt = PromptTemplate(template=llama_docs_template, input_variables=["context", "question"])
llama_doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt= llama_docs_prompt, document_variable_name="context", verbose=False)

llama_condense_template = """
[INST]Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: [/INST]"""
llama_condense_prompt = PromptTemplate(template=llama_condense_template, input_variables=["chat_history", "question"])
llama_question_generator_chain = LLMChain(llm=llm, prompt=llama_condense_prompt, verbose=False)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llama_v2_chain = ConversationalRetrievalChain(
    retriever=vector_index.as_retriever(search_kwargs={'k': 6}),
    question_generator=llama_question_generator_chain,
    combine_docs_chain=llama_doc_chain,
    memory=memory
)

print(llama_v2_chain({"question": "What models use human instructions?"}))

print(llama_v2_chain({"question": "Which are the advantage of each of these models?"}))

print(llama_v2_chain({"question": "What are the downsides of your last model suggested above ?"}))
