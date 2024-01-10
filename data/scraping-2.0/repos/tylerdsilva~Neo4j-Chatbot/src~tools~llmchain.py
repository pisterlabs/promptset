from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate

from llm import llm, memory

# Prompt template to define instructions
prompt = PromptTemplate(
    template="""
    Instructions:
    Do not answer any questions that do not relate to movies, actors, producers, followers, authors, reviewers, or directors
    Do not answer any questions using pre-trained knowledge, only use the information provided in the context.


    ChatHistory:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"]
    )

chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)