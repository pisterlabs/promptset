# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())

# from langchain.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import chainlit as cl

template = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
The answer must be given in Indonesian language.

Human: {input}
AI Assistant:"""

# Initialize chat
@cl.on_chat_start
def init():
    # Use LLM on local machine
    # chat_llm = ChatOpenAI(
    #     verbose=True
    # )
    chat_llm = ChatOllama(
        model="mistral",
        verbose=True
    )
    # Create prompt template
    prompt = PromptTemplate.from_template(template)

    # Create a chain
    chain = LLMChain(
        prompt=prompt,
        llm=chat_llm,
        verbose=True
    )

    # Store the chain into the session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(query: str):
    # Retrieve the chain
    chain = cl.user_session.get("chain")

    outputs = await chain.acall(
        query,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # print(f"outputs: {outputs}")
    # Send the response
    await cl.Message(
        content=outputs["text"]
    ).send()