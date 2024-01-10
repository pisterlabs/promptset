from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import chainlit as cl
import api

API_TOKEN = api.API_KEY

model_id = "gpt2-medium"
conv_model = HuggingFaceHub(huggingfacehub_api_token=API_TOKEN,
                            repo_id=model_id,
                            model_kwargs={"temperature": 1},)

template = """You are a helpful and friendly chatbot that makes conversation with the user.

{user_input}
"""

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["user_input"])
    llm_chain = LLMChain(prompt=prompt, llm=conv_model, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain synchronously in a different thread
    res = await cl.make_async(llm_chain)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()
    return llm_chain