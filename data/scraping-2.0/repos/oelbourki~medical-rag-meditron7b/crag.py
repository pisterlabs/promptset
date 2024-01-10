import chainlit as cl
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
import json
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import (
    GenerationConfig,
    pipeline,
)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.memory import ConversationBufferMemory
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
GPU_LAYERS=int(os.environ.get("GPU_LAYERS",35))
gpu_layers = round(GPU_LAYERS * 0.7)
LLAMA_VERBOSE=False
print("Running LLM Zephyr")
llm = LlamaCpp(model_path='zephyr-7b-beta.Q5_K_M.gguf',stream=True,temperature=0,
               callback_manager=callback_manager,
               max_new_tokens=256, context_window=4096, n_ctx=4096,n_batch=128, stop = ["\n\n"])

# llm = LlamaCpp(
#     model_path="meditron-7b-chat.Q4_K_M.gguf",
#     temperature=0.1,
#     max_tokens=1024,
#     top_p=1,
#     stop = ["\n\n"],
#     # callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )
print(llm, "LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db1")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

@cl.on_chat_start
async def start():
    print(prompt)
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    msg = cl.Message(content="Firing up the research info bot...")
    await msg.send()
    msg.content = "Hi, welcome to research info bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    # msg = cl.Message(content="")

    # async for chunk in chain.astream(message,
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)

    # await msg.send()
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    print(f"response: {res}")
    answer = res["result"]
    answer = answer.replace(".", ".\n")
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(str(sources))
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()

    
# @cl.on_message
# async def main(message: str):

# #     response = qa(query)
# #     print(response)
# #     answer = response['result']
# #     source_document = response['source_documents'][0].page_content
# #     doc = response['source_documents'][0].metadata['source']
# #     response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
# #     res = Response(response_data)
#     response = await cl.make_async(qa)(message)
#     sentences = response['answers'][0].answer.split('\n')

#     # Check if the last sentence doesn't end with '.', '?', or '!'
#     if sentences and not sentences[-1].strip().endswith(('.', '?', '!')):
#         # Remove the last sentence
#         sentences.pop()

#     result = '\n'.join(sentences[1:])
#     await cl.Message(author="Bot", content=result).send()
    
