# RUN: chainlit run model.py

# 01: CONFIGURE
#MODEL_ID="/work/u00cjz00/slurm_jobs/github/models/Llama-2-7b-chat-hf"
#MODEL_ID = "TheBloke/Llama-2-7b-Chat-GPTQ"
MODEL_ID = "/work/u00cjz00/slurm_jobs/github/models/Llama-2-7B-Chat-GPTQ"

DB_FAISS_PATH = 'vectorstore/db_faiss'

# 02: Load LIBRARY
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import transformers
import torch
from langchain.llms import HuggingFacePipeline
#from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#warnings.filterwarnings('ignore')

# 03: custom_prompt_template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# 04: Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 5}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm():
    # 04: LLM模型 GPTQ
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )
    llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
    
    return llm

# 05: QA Model Function
def qa_bot():
#    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                       model_kwargs={'device': 'cpu'})
    embeddings = HuggingFaceEmbeddings(model_name='/work/u00cjz00/slurm_jobs/github/models/embedding/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# 06: output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# 07: chainlit code
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
        answer += f"\n\n\n資料來源, Sources:" + str(sources)
    else:
        answer += "\n\n\nNo sources found"

    await cl.Message(content=answer).send()
    
