import os
import re
from pathlib import Path

import openai
import streamlit as st
from streamlit.components.v1 import html


def get_starting_convo():
    return [
        {
            "role": "system",
            "content": "You are an assistant that helps the user create and improve a web page in HTML, CSS, and JavaScript.",
        }
    ]


@st.cache_data(ttl=10, show_spinner=False)
def call_openai_api(conversation):
    st.spinner("The code is being generated...")
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=conversation, temperature=0.1
    )
    return response["choices"][0]["message"]["content"].strip()


def main():
    API_KEY = os.getenv("OPENAI_API_KEY")

    st.set_page_config(page_title="How do I get the jobseekers?", layout="wide")
    openai.api_key = API_KEY

    if "show_code" not in st.session_state:
        st.session_state.show_code = False
    if "messages" not in st.session_state:
        st.session_state.messages = get_starting_convo()

    st.title("Job Seeker Automated Application Filler")
    st.write("Enter application description below:")

    user_input = st.text_area("Type your content:", height=100)

    has_previously_generated = len(getattr(st.session_state, "messages", [])) > 1
    reset = not has_previously_generated or st.checkbox("Reset", value=False)

    # change text depending on whether the user has previously generated code

    if st.button(
        "Generate presentation"
        if reset
        else "Ask AI to modify the generated application"
    ):
        # set to starting conversation if reset
        messages = get_starting_convo() if reset else st.session_state.messages.copy()
        # decide prompt depending on whether the user has previously generated code
        if reset:
            messages.append(
                {
                    "role": "user",
                    "content": f"Create an HTML web page with accompanying CSS and JavaScript in a single HTML-file. Use suitable JS packages (linked from a CDN) where ever applicable. Generate this content from: {user_input}",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"Modify the previous website to accomodate the following:\n\n{user_input}\n\n Note that you should recreate the HTML, CSS, and JavaScript code from scratch in its entirety. The new code should be self-contained in a single HTML-file.",
                }
            )
        # get the AI's response
        try:
            output = call_openai_api(messages)
        except Exception as e:
            st.error(f"Error while generating code. {str(e)}")
            return

        # extract the code block
        pattern = r"```(\w*)\n([\s\S]*?)\n```"
        code_blocks = re.findall(pattern, output)

        # should be in a single file
        if len(code_blocks) != 1:
            st.error(
                "Something went wrong. Please try again or change the instructions."
            )
            return

        # append the assistant's response to the conversation
        messages.append({"role": "assistant", "content": output})

        # two groups are captured, the language and the code,
        # but only the code is needed
        st.session_state.code = code_blocks[0][1]
        st.session_state.output = output
        st.session_state.messages = messages

        # go from the beginning
        st.experimental_rerun()

    # display the generated html/css/javascript

    # html is from from streamlit.components.v1 import html
    if "code" in st.session_state and st.session_state.code is not None:
        st.header("Generated website")
        html(st.session_state.code, height=800, scrolling=True)

    # display the code and explanations
    if has_previously_generated and st.session_state.code is not None:
        # toggle between show and hide,
        # a placeholder (st.empty) is used to replace the
        # text of the button after it has been clicked
        show_hide_button = st.empty()
        get_show_code_text = (
            lambda: "Show code and explanations"
            if not st.session_state.show_code
            else "Hide code and explanations"
        )
        clicked = show_hide_button.button(get_show_code_text())
        if clicked:
            st.session_state.show_code = not st.session_state.show_code
            show_hide_button.button(get_show_code_text())
        if st.session_state.show_code:
            st.header("Code and explanations")
            st.markdown(st.session_state.output)


if __name__ == "__main__":
    main()
