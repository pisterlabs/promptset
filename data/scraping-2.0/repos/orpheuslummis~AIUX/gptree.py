# how to change the color if the submitted text is different from the original text?
# how to input api key as a modal 1st time?
# save logging to file


import hashlib
import json
import streamlit as st
import openai
from utils import RequestParams, new_logger, get_params_from_env, request

# defaults
TEMPERATURE = 0.6
MAX_TOKENS = 300
N = 4


def update_prompt():
    params = RequestParams(
        prompt=st.session_state.prompt,
        n=st.session_state.n,
        max_tokens=st.session_state.max_tokens,
        temperature=st.session_state.temperature,
    )
    log.info(params)
    with container_bottom:
        with st.spinner("Wait for it..."):
            results = request(params)
            # st.write(hash_dict(data))
            for i, r in enumerate(results):
                st.markdown(r)
                if i < len(results) - 1:
                    st.write("–" * 100)


def hash_dict(d):
    data = json.dumps(d, sort_keys=True)
    hh = hashlib.sha256(data.encode()).hexdigest()
    return hh


if __name__ == "__main__":
    log = new_logger("gptree")
    params = get_params_from_env()
    if params["apikey"] is None:
        st.error("Please set OPENAI_API_KEY environment variable.")
    openai.api_key = params["apikey"]

    st.set_page_config(layout="wide")
    container_top = st.container()
    container_bottom = st.container()

    with container_top:
        st.text_area("Prompt", key="prompt")
        if st.button("Submit"):
            update_prompt()

        with st.expander("Advanced"):
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=TEMPERATURE,
                step=0.1,
                key="temperature",
            )
            st.slider(
                "Max Tokens",
                min_value=1,
                max_value=1000,
                value=MAX_TOKENS,
                step=1,
                key="max_tokens",
            )
            st.slider("N", min_value=1, max_value=10, value=N, step=1, key="n")
