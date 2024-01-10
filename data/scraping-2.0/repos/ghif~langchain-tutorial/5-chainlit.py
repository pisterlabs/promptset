from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

import chainlit as cl
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())


# Initialize chat
@cl.on_chat_start
def init():
    """
    Model
    """
    chat_llm = ChatOpenAI(
        temperature=0.0,
        streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()]
    )


    # Chain1: input=Review, output=Indonesian_Review
    prompt1 = ChatPromptTemplate.from_template(
        """
        Translate the following review to Indonesian:
        {review}
        """
    )
    chain1 = LLMChain(
        llm=chat_llm,
        prompt=prompt1,
        output_key="indonesian_review",
        verbose=True
    )

    # Chain2: input=Review, output=Sentiment
    prompt2 = ChatPromptTemplate.from_template(
        """
        Is the following review positive or negative?
        {review}

        Answer with either "positive" or "negative"
        """
    )
    chain2 = LLMChain(
        llm=chat_llm,
        prompt=prompt2,
        output_key="sentiment",
        verbose=True
    )

    # Chain3: input=Indonesian_Review, output=Summary
    prompt3 = ChatPromptTemplate.from_template(
        """
        Summarize the following Indonesian review in 1 sentence:
        {indonesian_review}
        """
    )
    chain3 = LLMChain(
        llm=chat_llm,
        prompt=prompt3,
        output_key="summary",
        verbose=True
    )


    overall_seq_chain = SequentialChain(
        chains=[chain1, chain2, chain3],
        input_variables=["review"],
        output_variables=["indonesian_review", "sentiment", "summary"],
        verbose=True
    )
    
    cl.user_session.set("chain", overall_seq_chain)

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")

    # Infer from the chain
    outputs = await chain.acall(
        message,
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

    # Post-processing
    res = f"""
    Summary: {outputs["summary"]}
    
    Sentiment: {outputs["sentiment"]}

    Indonesian Review: {outputs["indonesian_review"]}
    """

    # Send the response 
    await cl.Message(
        content=res
    ).send()