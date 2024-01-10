from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the information you have to list the causes and precautionary measures to the symptoms of their child given by the user and do not suggest any causes that might be fatal to patient

Use the following symptoms: {question}
By searching the following context: {context}

Make sure you 
Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    # string prompt
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                        memory=memory,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                    #    return_source_documents=False,
                                    #    chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    model="TheBloke/zephyr-7B-beta-GGUF"
    model_path="zephyr-7b-beta.Q4_K_M.gguf"
    model_type="mistral"


    config={
    'max_length':2048,
    "repetition_penalty":0.5,
    "temperature":0.6,
    'top_k':50,
    "top_p":0.9,
    # 'gpu_layers':50,
    "stream":False,
    'threads':os.cpu_count()
    }

    llm_init=CTransformers(
    model=model_path,
    model_type=model_type,
    lib="avx2",
    **config
    )
    return llm_init

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response
chat_history=[]

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to MediBuddy. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)
    # cl.user_session.set("chat_history",[])

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    # chat_history=cl.user_session.get("chat_history") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(str(message.content),chat_history, callbacks=[cb])
    #print(res)
    answer = res["answer"]
    chat_history.append((message,answer))
    #print(res)

    await cl.Message(content=answer).send()
