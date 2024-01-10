import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import chainlit as cl
import textwrap

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# --------------------------------------------------------------

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)

# --------------------------------------------------------------
# template = """Question: {question}

# Answer: Let's think step by step."""

# template = """Question: {question}
# Answer: """


template = """Answer the following question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm,verbose=True)


@cl.on_message
async def main(message: str):
    # Your custom logic goes here...
    
    response = llm_chain.run(message)
    wrapped_text = textwrap.fill(
    response, width=100, break_long_words=False, replace_whitespace=False)
    # Send a response back to the user
    await cl.Message(
        content=f"{wrapped_text}",
    ).send()
