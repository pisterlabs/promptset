from config import get_OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
llm_model = "gpt-3.5-turbo-1106"
open_ai = OpenAI(temperature = 0.0)

def generate_story(location, name, language):
    template = """
        As a childrens book author, write a simple and short (90 words) story lullaby based on the location
        {location}
        and the main character
        {name}

        STORY:
    """

    prompt = PromptTemplate(
        input_variables=["location", "name"],
        template=template,
    )

    chain_story = LLMChain(llm=open_ai, prompt=prompt, output_key="story", verbose=True)

    # SequentialChain
    translation_template = """
        Translate the {story} to {language}.

        Make sure the translation is simple and fun to read for children.

        TRANSLATION:
    """

    prompt_translation = PromptTemplate(
        input_variables=["story", "language"],
        template=translation_template,
    )

    chain_translation = LLMChain(llm=open_ai, prompt=prompt_translation, output_key="translated")

    overall_chain = SequentialChain(
        chains=[chain_story, chain_translation],
        input_variables=["location", "name", "language"],
        output_variables=["story", "translated"])

    response = overall_chain({"location": location, "name": name, "language": language })
    return response

def main():
    st.set_page_config(page_title="Story Generator", page_icon="📚", layout="centered")
    st.title("AI 📖 Story Generator")
    st.header("Get started...")
    
    location_input = st.text_input(label="Enter a location")
    character_input = st.text_input(label="Enter a name")
    language_input = st.text_input(label="Enter a language")
    
    submit_button = st.button("Generate Story")
    if location_input and character_input and language_input:
        if submit_button:
            with st.spinner("Generating story..."):
                response = generate_story(location=location_input, name=character_input, language=language_input)
                with st.expander("English Version"):
                    st.write(response['story'])
                with st.expander(f"{language_input} Version"):
                    st.write(response['translated'])
                st.success("Story generated!")

    
    pass



# Invoke Main function
if __name__ == "__main__":
    main()