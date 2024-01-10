import streamlit as st


def run():
    st.set_page_config(
        page_title="GPT-4V Demos",
        page_icon="🤖",
        initial_sidebar_state="expanded",
    )

    try:
        # Import OpenAI API key if it exists in project secrets
        st.session_state.api_key = st.secrets.OPENAI_API_KEY
    except:
        # Otherwise, prompt for OpenAI API key
        st.session_state.api_key = st.sidebar.text_input(
            "OpenAI API key",
            st.session_state.api_key if "api_key" in st.session_state else "",
            type="password",
        )

    if st.session_state.api_key == "":
        st.sidebar.caption(":red[An OpenAI API key is required to run the tests.]")

    st.write("# GPT-4V Demos")
    st.write("\n")
    st.info(
        "This mobile-friendly web app provides some basic demos to test the vision capabilities of GPT-4V."
    )
    st.info("Open them from the sidebar!", icon="↖️")
    st.caption(
        """This project is licensed under the terms of the MIT license.
        [View the source code](https://github.com/logicalroot/gpt-4v-demos)."""
    )
    st.write("\n")
    st.markdown(
        """
        ### About GPT-4V\n
        [OpenAI announcement](https://openai.com/blog/new-models-and-developer-products-announced-at-devday)\n
        [OpenAI research](https://openai.com/research/gpt-4v-system-card)\n
        [OpenAI docs](https://platform.openai.com/docs/guides/vision)\n
        """
    )


if __name__ == "__main__":
    run()
