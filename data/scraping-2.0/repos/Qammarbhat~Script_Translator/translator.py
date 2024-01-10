from langchain import PromptTemplate
import os
from langchain.llms import OpenAI

# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "Your API KEY"

# Create an instance of the OpenAI language model (LLM) using the gpt-3.5-turbo model
llm = OpenAI(model_name="gpt-3.5-turbo")

# Define a function for translating text with localization
def text_translator(input_text, input_target_language, input_country_names):
    # Define a template for the translation prompt using PromptTemplate
    translate_prompt = PromptTemplate(
        input_variables=["input_text", "target_language", "country_names"],
        template="""
        You are an advanced translator specializing in localizing scripts. Your task is to translate the following script into {target_language} while replacing any names with culturally appropriate names for {country_names}. Make sure to maintain the context and tone of the script.

        Here is the script for localization:

        {input_text}

        As you replace names, consider the cultural and linguistic context of {country_names}. Deliver a translation that resonates with the audience in {country_names}.
        """
    )

    # Calculate the maximum input length for tokenization
    max_input_length = 4096 - len(
        translate_prompt.format(
            input_text="",
            target_language=input_target_language,
            country_names=input_country_names,
        )
    )

    translated_chunks = []  # Initialize a list to store translated chunks
    start_idx = 0  # Initialize the starting index for chunking

    # Split the input text into chunks and translate each chunk
    while start_idx < len(input_text):
        # Calculate end index of the current chunk
        end_idx = min(start_idx + max_input_length, len(input_text))

        # Extract the current chunk of input text
        chunk = input_text[start_idx:end_idx]

        # Generate translated text for the chunk using the translation prompt
        translated_chunk = llm(
            translate_prompt.format(
                input_text=chunk,
                target_language=input_target_language,
                country_names=input_country_names,
            )
        )

        # Append the translated chunk to the list
        translated_chunks.append(translated_chunk)

        # Move to the next chunk
        start_idx = end_idx

    # Join the translated chunks to get the complete translated text
    translated_text = " ".join(translated_chunks)
    return translated_text
