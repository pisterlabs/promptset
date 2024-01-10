import chainlit as cl

import os
huggingfacehub_api_token = os.environ['HUGGINGFACE_TOKEN']

from langchain import HuggingFaceHub, PromptTemplate, LLMChain

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

template = """
You are a helpful, respectful and honest assistant.

{question}

"""


@cl.on_chat_start
async def on_start():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    cl.user_session.set("chain",llm_chain)

@cl.on_message
async def on_message(message: cl.Message):
    llm_chain = cl.user_session.get("chain") # type: LLMChain

    result = await llm_chain.arun(
        question = message.content, callbacks = [cl.LangchainCallbackHandler()]
    )

    await cl.Message(content=result).send()