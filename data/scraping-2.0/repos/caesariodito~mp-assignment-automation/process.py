import openai
import keyword_generator as key_gen
import streamlit as st

# api_key = e.OPENAI_API_KEY


@st.experimental_memo
def solve(text: str, api_key: str, lang='english') -> list[str, str]:
    """This function is used to solve the question and sent request to openai api

    Args:
        text (str): raw text input
        api_key (str): api secret key to send request to OpenAI API 

    Returns:
        list[str, list]: The result contains response from openai api and keywords is a list containing keywords that is used in the prompt parameter api
    """
    # keywords = key_gen.get_keywords(text)
    keywords = key_gen.get_keywords(text, api_key=api_key, lang=lang)

    DESIGN_PROMPT_2 = f"""
    Provide me detailed and related insight between keywords and summarize it. Provide all output using {lang} language. 
    """
    prompt2 = f"""
    The keywords are:
    {keywords}
    """

    final_prompt = DESIGN_PROMPT_2 + prompt2

    openai.api_key = api_key

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=final_prompt,
        temperature=0.3,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0
    )

    result = response['choices'][0]['text']

    return result, keywords
