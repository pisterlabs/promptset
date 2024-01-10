import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain


def apply_langchain(text_input):
    llm = OpenAI(temperature=.7, openai_api_key="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", max_tokens=2049)
    template = """Select which category the post belong.
    Post: 
    {post} 
    """
    prompt_template = PromptTemplate(input_variables=["post"], template=template)
    post_chain = LLMChain(llm=llm, prompt=prompt_template)
    print(post_chain.run(text_input))
    return post_chain.run(text_input)
    # This is an LLMChain to write a review of a play given a synopsis.
    # llm = OpenAI(temperature=.7, openai_api_key="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", max_tokens=2049)
    # template = """
    # You need to select one category memory loss, driving, eating"""
    # prompt_template = PromptTemplate(input_variables=["food"], template=template)
    # category_chain = LLMChain(llm=llm, prompt=prompt_template)
    #
    # overall_chain = SimpleSequentialChain(chains=[post_chain, category_chain], verbose=True)
    # review = overall_chain.run(text_input)
    # print(review)

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 = st.columns(2)

with col1:

    text_input = st.text_input(
        "Input the post",
        "",
        key="placeholder",
    )
    # print("place holder is", text_input)

with col2:
    st.caption('This is a string that explains something above.')
    st.caption(apply_langchain(text_input))





