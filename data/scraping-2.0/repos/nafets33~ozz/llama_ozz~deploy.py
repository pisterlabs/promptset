from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from datetime import datetime
DB_FAISS_PATH = "vectorstores/db_fiass/"

custom_prompt_template='''Use the following pieces of information to answer the users question. 
If you don't know the answer, please just say that you don't know the answer. Don't make up an answer.

Context:{context}
question:{question}

Only returns the helpful anser below and nothing else.
Helpful answer
'''

print("dones")

def set_custom_prompt():
 '''
 Prompt template for QA retrieval for each vector store
 '''
 prompt =PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

 return prompt


def load_llm():
 llm = CTransformers(
  model='llama-2-7b-chat.ggmlv3.q8_0.bin',
  model_type='llama',
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

def qa_bot(embeddings):
#  embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#   model_kwargs={'device':'cpu'})
 db = FAISS.load_local(DB_FAISS_PATH,embeddings)
 llm=load_llm()
 qa_prompt=set_custom_prompt()
 qa = retrieval_qa_chain(llm,qa_prompt,db)
 return qa 


embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
def final_result(query, embeddings):
 s = datetime.now()
 query = "Tell me about the rain walk in detail. Provide a step by step process on how to experience the walk and what I will learn on the walk. Please also create questions a 7 year old should ask their parents what they have learned about the walk"
 qa_result=qa_bot(embeddings)
 response=qa_result({'query':query})
 e = datetime.now()
 print((e-s).total_seconds())
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
 res=await chain.acall(message, callbacks=[cb])
 answer=res["result"]
 sources=res["source_documents"]

 if sources:
  answer+=f"\nSources: "+str(str(sources))
 else:
  answer+=f"\nNo Sources found"

 await cl.Message(content=answer).send() 