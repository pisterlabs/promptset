import streamlit as st
from validate import validate_url
import requests
from bs4 import BeautifulSoup
from langchain.schema import (
    HumanMessage,
)
from langchain.callbacks import get_openai_callback


def handle_web_summarize(llm):
    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("URL is not valid")
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content, 1000)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None
    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)

    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url


def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            # fetch text from main (change the below code to filter page)
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except Exception as e:
        st.write(f"something wrong\n\n{e}")
        return None


def build_prompt(content, n_chars=300):
    return f"""以下はWebページのコンテンツである。内容を{n_chars}文字程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてください！
"""


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost
