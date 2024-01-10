from langchain.llms import HuggingFacePipeline, CTransformers
import langchain
from ingest import load_db
from langchain.cache import InMemoryCache
from langchain.schema import prompt
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain import PromptTemplate
import chainlit as cl
DB_FAISS_PATH = 'vectorstoredb/db_faiss'


langchain.llm_cache = InMemoryCache()

PROMPT_TEMPLATE = '''
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Do provide only helpful answers

Helpful answer:
'''
handler = StdOutCallbackHandler()
def set_custom_prompt():
    input_variables = ['context', 'question']
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=input_variables)
    return prompt

def load_retriever():
    return load_db()

def load_llm():
    llm = CTransformers(model="models/mistral-7b-instruct-v0.1.Q2_K.gguf", config={'max_new_tokens': 128, 'temperature': 0.01})
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = db,
    verbose=True,
    callbacks=[handler],
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
    )  
    return qa_chain

def qa_bot():
    db = load_retriever()
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()



