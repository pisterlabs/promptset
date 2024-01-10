from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chainlit as cl

VECTORSTORE_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question. If you don't know the answer, 
                            just say that you don't know, don't try to make up an answer.

                            context:{context}
                            Question:{question}

                            Only return the helpful answer below and nothing else.
                            Helpful answer:
                            """


def set_custom_prompt():
    """
    Prompt template for QA retrieval from vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=["context","question"])
    return prompt


# load fundational model
def load_llm():
    # local the localy downloaded foundational model
    llm = CTransformers(#model="G:\\Python Practice\\FoundationModels\\llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model="TheBloke/Llama-2-7B-Chat-GGML",
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.3)
    return llm


# Retrieval QA chain
def retrieval_qa_chain(llm, prompt, kb):
    va_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever=kb.as_retriever(search_kwargs={'k': 2}),
                                            return_source_documents=True,
                                            chain_type_kwargs={'prompt': prompt}
                                            )

    return va_chain
    

#Medicine Question Answer Virtual Assistat
def virtualassistant():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    print("Embeddings completed")
    kb = faiss.FAISS.load_local(VECTORSTORE_PATH, embeddings)
    print("Loading LLM")
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    va = retrieval_qa_chain(llm, qa_prompt, kb)
    return va


# create final output
def response(query):
    va_result = virtualassistant()
    response_ = va_result({"query":query})
    return response_


##################################################################################################################################

@cl.on_chat_start
async def start_chat():
    chain = virtualassistant()
    message = cl.Message(content="Starting the virtual Assistant...")
    await message.send()
    message.content = "Hi, Welcome to Medical Bot. What is your query?"
    await message.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message:cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL","ANSWER"]
    )
    cb.answer_reached = True
    result = await chain.acall(message.content, callbacks=[cb])
    answer = result["result"]
    sources = result["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo sources found"


    await cl.Message(content=answer).send()












