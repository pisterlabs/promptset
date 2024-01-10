from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
import chainlit as cl
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)

template = """Question: {question}

Answer: Let's think step by step."""
CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])

model_file_path = ""

llm = LlamaCpp(
    model_path=model_file_path,
    n_ctx=6000,
    n_gpu_layers=512,
    n_batch=30,
    callback_manager=CallbackManager,
    temperature=0.9,
    max_tokens=4095,
    n_parts=1,
    verbose=0

)

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()
