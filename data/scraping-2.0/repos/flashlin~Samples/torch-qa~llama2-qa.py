"""
chainlit run .\langchain_llama2.py -w
"""
import asyncio

from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "models/db_faiss"
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


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=[
        'context', 'question'])
    return prompt


def load_llm():
    llm = CTransformers(model=f"models/{MODEL_PTH_NAME}",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.5)
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


async def main():
    chain = qa_bot()
    while True:
        user_input = input("query: ")
        if user_input == "q":
            break
        query = user_input
        res = await chain.acall(query)
        answer = res["result"]
        sources = res["source_documents"]
        if sources:
            answer += f"\nSources:" + str(sources[0].metadata['source'])
        else:
            answer += f"\nNo Sources Found"
        print(f"{answer}")


if __name__ == '__main__':
    asyncio.run(main())
