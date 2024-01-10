
import pdb
import sys
sys.path.append("..")
from config import set_environment
set_environment()

from langchain.chains import LLMCheckerChain
from langchain.llms import OpenAI
from langchain import PromptTemplate, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
# from langchain_decorators import llm_prompt
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.agents import (
    AgentExecutor, AgentType, initialize_agent, load_tools
)
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler


def checkerchain():
    llm = OpenAI(temperature=0.7)
    text = "What type of mammal lays the biggest eggs?"
    checker_chain = LLMCheckerChain.from_llm(llm, verbose=True)
    print(checker_chain.run(text))


def prompting_func1():
    prompt = """
        Summarize this text in one sentence: {text}
    """
    llm = OpenAI()
    text = """Machine learning (ML) is the study of computer algorithms that improve 
            automatically through experience. It is seen as a subset of artificial intelligence. 
            Machine learning algorithms build a model based on sample data, known as 
            training data, in order to make predictions or decisions without being explicitly 
            programmed to do so. Machine learning algorithms are used in a wide variety of 
            applications, such as email filtering and computer vision, where it is difficult or 
            infeasible to develop a conventional algorithm for effectively performing the task."""
    summary = llm(prompt.format(text=text))
    print(summary)


def lc_StrOutputParser():
    llm = OpenAI()
    prompt = PromptTemplate.from_template(
    "Summarize this text: {text}?"
    )
    text = "this is a text"
    runnable = prompt | llm | StrOutputParser()
    summary = runnable.invoke({"text": text})
    print(summary)


# @llm_prompt
# def summarize(text:str, length="short") -> str:
#     """
#     Summarize this text in {length} length:
#     {text}
#     """
#     return


def article_read():
    template = """Article: { text }
        You will generate increasingly concise, entity-dense summaries of the
        above article.
        Repeat the following 2 steps 5 times.
        Step 1. Identify 1-3 informative entities (";" delimited) from the article
        which are missing from the previously generated summary.
        Step 2. Write a new, denser summary of identical length which covers every
        entity and detail from the previous summary plus the missing entities.
        A missing entity is:
        - relevant to the main story,
        - specific yet concise (5 words or fewer),
        - novel (not in the previous summary),
        - faithful (present in the article),
        - anywhere (can be located anywhere in the article).
        Guidelines:
        - The first summary should be long (4-5 sentences, ~80 words) yet highly
        non-specific, containing little information beyond the entities marked
        as missing. Use overly verbose language and fillers (e.g., "this article
        discusses") to reach ~80 words.
        - Make every word count: rewrite the previous summary to improve flow and
        make space for additional entities.
        - Make space with fusion, compression, and removal of uninformative
        phrases like "the article discusses".
        - The summaries should become highly dense and concise yet self-contained,
        i.e., easily understood without the article.
        - Missing entities can appear anywhere in the new summary.
        - Never drop entities from the previous summary. If space cannot be made,
        add fewer new entities.
        Remember, use the exact same number of words for each summary.
        Answer in JSON. The JSON should be a list (length 5) of dictionaries whose
        keys are "Missing_Entities" and "Denser_Summary".
        """
    

def map_reduce():
    pdf_file_path = "<pdf_file_path>"
    pdf_loader = PyPDFLoader(pdf_file_path)
    docs = pdf_loader.load_and_split()
    llm = OpenAI()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    chain.run(docs)


def callback_func():
    llm_chain = PromptTemplate.from_template("Tell me a joke about {topic}!")
    with get_openai_callback() as cb:
        response = llm_chain.invoke(dict(topic="light bulbs"))
        print(response)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    
    input_list = [
        {"product": "socks"},
        {"product": "computer"},
        {"product": "shoes"}
    ]
    print(llm_chain.generate(input_list))

    # {
    #     "model": "gpt-3.5-turbo-0613",
    #     "object": "chat.completion",
    #     "usage": {
    #         "completion_tokens": 17,
    #         "prompt_tokens": 57,
    #         "total_tokens": 74
    #     }
    # }


def load_agent() -> AgentExecutor:
    llm = ChatOpenAI(temperature=0, streaming=True)
    # DuckDuckGoSearchRun, wolfram alpha, arxiv search, wikipedia
    # TODO: try wolfram-alpha!
    tools = load_tools(
        # tool_names=["ddg-search", "wolfram-alpha", "arxiv", "wikipedia"],
        tool_names=["ddg-search", "arxiv", "wikipedia"],
        llm=llm
    )
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )


def streamlit_app():
    chain = load_agent()
    st_callback = StreamlitCallbackHandler(st.container())
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = chain.run(prompt, callbacks=[st_callback])
            st.write(response)

# checkerchain()
# prompting_func1()
# summary = summarize(text="let me tell you a boring story from when I was young...")
# lc_StrOutputParser()
# article_read()
# map_reduce()
# callback_func()
# load_agent()
# streamlit_app()