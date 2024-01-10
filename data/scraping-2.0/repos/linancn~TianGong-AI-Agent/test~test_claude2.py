import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def main_chain():
    langchain_verbose = st.secrets["langchain_verbose"]
    openrouter_api_key = st.secrets["openrouter_api_key"]
    openrouter_api_base = st.secrets["openrouter_api_base"]

    selected_model = "anthropic/claude-2"
    # selected_model = "openai/gpt-3.5-turbo-16k"
    # selected_model = "openai/gpt-4-32k"
    # selected_model = "meta-llama/llama-2-70b-chat"

    llm_chat = ChatOpenAI(
        model_name=selected_model,
        # temperature=0,
        streaming=True,
        # verbose=langchain_verbose,
        openai_api_key=openrouter_api_key,
        openai_api_base=openrouter_api_base,
        headers={"HTTP-Referer": "http://localhost"},
        # callbacks=[],
    )

    prompt_template = "Tell me about {thing}"
    prompt = PromptTemplate(input_variables=["thing"], template=prompt_template)
    chain = LLMChain(
        llm=llm_chat,
        prompt=prompt,
        verbose=langchain_verbose,
    )
    return chain


chain = main_chain()

response = chain.run({"thing": "large language model"})

print(response)
