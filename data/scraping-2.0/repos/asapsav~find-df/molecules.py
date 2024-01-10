import streamlit as st
import json
import dotenv
import os
import requests
dotenv.load_dotenv()
import openai
import networkx as nx
import matplotlib.pyplot as plt


YDC_API_KEY = os.getenv("YDC_API_KEY")
YDC_API_KEY_RAG = os.getenv("YDC_API_KEY_RAG")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4-1106-preview"

def get_ai_snippets_for_query(query):
    headers = {"X-API-Key": YDC_API_KEY}
    params = {"query": query}
    return requests.get(
        f"https://api.ydc-index.io/search?query={query}",
        params=params,
        headers=headers,
    ).json()

def get_you_rag(query):
    headers = {"X-API-Key": YDC_API_KEY_RAG}
    params = {"query": query}

    return requests.get(
        f"https://api.ydc-index.io/rag?query={query}",
        params=params,
        headers=headers,
    ).json()


def get_openai_completions(prompt):
    messages = [
        {"role": "system", "content": "You are a helpfull assistant that is an exper biologist and chemo informatitian"}
    ]
    messages.append({"role": "user", "content": prompt})
    
    completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=messages
    )
    
    assistant_message = completion.choices[0].message['content']
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


# Main application
def main():
    st.title("Use you.com API")
    st.markdown("""
                * BARTsmiles Paper: https://arxiv.org/pdf/2211.16349.pdf \n
                * S3 Bucket with molecules: https://registry.opendata.aws/usearch-molecules/ \n
                """)

    # Input section
    query_input = st.text_input("Type your query here")
    
    if st.button("Get info"):
        if query_input:
            output = get_ai_snippets_for_query(f"{query_input}")
            st.success("finished YOU.COM Search, getting GPT-4 summary")
            st.subheader(f"YOU.COM Search Result: {len(output)} hits")
            for hit in output["hits"]:
                with st.expander(hit["title"]):
                    st.markdown(hit["description"])
                    if st.checkbox("Show snippets", key=hit["title"]):
                        for snippet in hit["snippets"]:
                            st.markdown(snippet)

            #summary = get_openai_completions(f"Sumamrise this:{output}")
            #st.subheader("YOU.COM Search Summary")
            #st.markdown(f"{summary}")

            st.subheader("YOU.COM RAG Result")
            ydc_rag_result = get_you_rag(f"{query_input}")
            st.json(ydc_rag_result)

            st.subheader("VDB Search Results")

            st.subheader("Smth else")

        else:
            st.error("Please input search query.")

if __name__ == "__main__":
    main()
