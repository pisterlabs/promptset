#pip install langchain llama-cpp-python chainlit

# https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF/tree/main

import chainlit as cl

from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="d:\\python code\\models\\airoboros-l2-7b-2.1.Q2_K.gguf",
)

template = """Question: {question}
Answer: Let's think step by step."""

# How to react when connection is established
@cl.on_chat_start
def main():
    # Instantiate the chain for that user session

    prompt = PromptTemplate(template=template, input_variables=["question"])

    #Invoked every time message is sent
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Sets the chain in the user session
    # user_session is a dictionary with set and get methods
    # key : "llm_chain" , value :  llm_chain
    cl.user_session.set("llm_chain", llm_chain)


#  How to react each time a user sends a message.
#  Then, we send back the answer to the UI with the Message class.
@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session by Passing the key
    llm_chain = cl.user_session.get("llm_chain")

    # Call the chain asynchronously - Use await keyword to call asynchronous function
    # Components communicating : Chainlit UI and langchain agent
    # When running your LangChain agent, it is essential to provide the appropriate callback handler.
    # acall is a function in Chainlit LangChain that allows you to call an asynchronous function with a message as input.
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print("Result : ",res)

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.

    # Use Message class to send, stream, edit or remove messages in the chatbot user interface.
    await cl.Message(content=res["text"]).send()
