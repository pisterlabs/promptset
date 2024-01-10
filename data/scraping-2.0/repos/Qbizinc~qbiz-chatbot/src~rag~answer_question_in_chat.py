import streamlit as st
import os
import weaviate
import json
from openai import OpenAI


def run_app():

    st.title("QBiz Chatbot")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Ask Me Anything about QBiz internal Documentation!"}
        ]

    st.write('---')

    weaviate_client = weaviate.Client(
        url = "https://weviate-cluster-xom22sd6.weaviate.network",
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEVIATE_API_KEY"]),
        additional_headers = {
            "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]
        }
    )

    openai_client = OpenAI(
       api_key=os.environ["OPENAI_API_KEY"],
    )

    if query_string := st.chat_input("Your Question"):
        st.session_state.messages.append({"role": "user",
                                          "content": query_string})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            


    


    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                weaviate_response = (
                    weaviate_client.query
                    .get("Text_block", ["text_block"])
                    .with_near_text({"concepts": [query_string]})
                    .with_limit(3)
                    .do()
                )

                additional_context = [w["text_block"] for w in  weaviate_response["data"]["Get"]["Text_block"]]
                joined_texts = '\n\n'.join([a for a in additional_context])

                system_prompt = f"""I want you to answer a user query using only the following source texts as references: {joined_texts}.
                Include the referenced document in the answer."""
                completion = openai_client.chat.completions.create(
                              model="gpt-3.5-turbo",
                              messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": query_string}
                              ]
                            )
                st.write(completion.choices[0].message.content)
                message = {"role": "assistant", "content": completion.choices[0].message}
                st.session_state.messages.append(message)  # Add response to message history



if __name__ == "__main__":
    run_app()
