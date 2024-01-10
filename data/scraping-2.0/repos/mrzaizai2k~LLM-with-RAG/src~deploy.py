import sys
sys.path.append("")

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import torch
from src.utils import *

DB_FAISS_PATH = "vectorstores/db_faiss/"

custom_prompt_template='''Use the following pieces of information to answer the users question. 
If you don't know the answer, please just say that you don't know the answer. Don't make up an answer.

Context:{context}
question:{question}

Only returns the helpful anser below and nothing else.
Helpful answer
'''

device =  take_device()

def set_custom_prompt():
    '''
    Prompt template for QA retrieval for each vector store
    '''
    prompt =PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

    return prompt

# def load_llm():
#     llm = CTransformers(
#     model='llama-2-7b-chat.ggmlv3.q4_1.bin',
#     model_type='llama',
#     max_new_tokens=512,
#     temperature=0.5
#     )
#     return llm

def load_llm():
    llm = CTransformers(
        model='TheBloke/Llama-2-7B-Chat-GGML', 
        model_file='llama-2-7b-chat.ggmlv3.q4_1.bin',
        max_new_tokens=512,
        temperature=0.5
        )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':prompt  }
    )
    return qa_chain

def qa_bot():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':device})
    db = FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm=load_llm()
    qa_prompt=set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,db)
    return qa 


def final_result(query):
    qa_result=qa_bot()
    response=qa_result({'query':query})
    return response 

## chainlit here
@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Firing up the company info bot...")
    await msg.send()
    msg.content= "Hi, welcome to company info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain",chain)


@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
    stream_final_answer=True, answer_prefix_tokens=["FINAL","ANSWER"]
    )
    cb.ansert_reached=True
    # res=await chain.acall(message, callbacks=[cb])
    res=await chain.acall(message.content, callbacks=[cb])
    answer=res["result"]
    sources=res["source_documents"]

    if sources:
        answer+=f"\nSources: "+str(str(sources))
    else:
        answer+=f"\nNo Sources found"

    await cl.Message(content=answer).send() 