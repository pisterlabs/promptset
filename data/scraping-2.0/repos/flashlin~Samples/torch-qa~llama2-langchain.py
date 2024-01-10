"""
chainlit run .\langchain_llama2.py -w
"""
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "models/db_faiss"
MODEL_PTH_NAME = "llama-2-13b-chat.ggmlv3.q6_K.bin"  # 尚未下載
MODEL_PTH_NAME = "llama-2-13b-chat.ggmlv3.q8_0.bin"
MODEL_PTH_NAME = "llama-2-7b-chat.ggmlv3.q8_0.bin"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just sat that you don't know the answer,
don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
"""

custom_prompt_template = """SYSTEM: You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible. 
Please ensure that your responses are positive in document. 
If a question does not make any sense, or is not factually coherent, 
explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.
please just sat that you don't know the answer, don't try to make up an answer.

USER: {question}
ASSISTANT: {context}
"""


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
        'context', 'question'])
    return prompt


def load_llm():
    llm = CTransformers(model=f"models/{MODEL_PTH_NAME}",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)
    return llm


def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


### Chainlit ###
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Bot, What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    length = len(answer)
    answer = answer[:length//2]
    sources = res["source_documents"]
    filenames = []
    for doc in sources:
        filenames.append(doc.metadata)
    if sources:
        answer += f"\nSources:" + str(filenames)
    else:
        answer += f"\nNo Sources Found"
    await cl.Message(content=answer).send()
