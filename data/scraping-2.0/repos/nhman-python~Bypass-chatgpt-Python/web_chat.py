import streamlit as st
import openai


@st.cache_data
def complete_text(prompt_user, token, engine):
    if prompt_user:
        prompt_user = f"Act as a professional AI automation system capable of providing concise and accurate answers " \
                      f"to various questions. Focus on understanding the problem or question at hand and provide " \
                      f"brief responses without excessive descriptions. Prompt: {prompt_user}"

        response = openai.Completion.create(
            engine=engine,
            prompt=prompt_user,
            max_tokens=token,
            temperature=0.7,
            n=1,
            stop=None
        )
        if response.choices and len(response.choices) > 0:
            return response.choices[0].text
        else:
            return "[AI]: Unable to generate a response."
    else:
        return "[AI]: Please type something."


class ApiError(Exception):
    pass


def main():
    st.set_page_config(
        page_title="gpt python tip israel",
        page_icon=":robot:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Set the theme
    theme = """
    <style>
    body {
        color: #e4d0d0;
        background-color: #3d4b83;
    }
    .st-bb {
        background-color: #7d5e5e;
    }
    .st-dd, .st-de, .st-du {
        color: #e4d0d0;
    }
    </style>
    """
    st.markdown(theme, unsafe_allow_html=True)

    try:
        st.title("OpenAI Conversation")
        st.markdown('## ask me something:')
        prompt = st.text_input('', key='input_prompt')
        tokens = st.slider('Number of Tokens', min_value=10, max_value=500, value=100)
        engine = st.selectbox('Engine', ['text-davinci-003', 'text-davinci-002'])
        api_key = st.text_input('OpenAI API Key', type='password')
        if prompt:
            if api_key:
                openai.api_key = api_key
                st.write(complete_text(prompt, tokens, engine))
            else:
                raise openai.error.APIError
    except openai.error.APIError as error_api:
        st.write('[ERROR]: You missed the API key. Please make sure to set a valid API key. You can obtain an OpenAI '
                 'API key by visiting this link: https://platform.openai.com/account/api-keys.')
    except openai.error.InvalidRequestError as invalid:
        st.write(f'[ERROR]: Something went wrong. Please try again.\nError message: {invalid}')


if __name__ == "__main__":
    main()
