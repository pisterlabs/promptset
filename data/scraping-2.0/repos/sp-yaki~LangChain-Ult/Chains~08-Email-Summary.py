import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

spanish_email = open('../some_data/spanish_customer_email.txt').read()


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain,SequentialChain

def translate_and_summarize(email):
    """
    Translates an email written in a detected language to English and generates a summary.

    Args:
        email (str): The email to be processed and translated.

    Returns:
        dict: A dictionary containing the following keys:
            - 'language': The language the email was written in.
            - 'translated_email': The translated version of the email in English.
            - 'summary': A short summary of the translated email.

    Raises:
        Exception: If any error occurs during the LLM chain execution.

    Example:
        email = "Hola, ¿cómo estás? Espero que todo vaya bien."
        result = translate_and_summarize(email)
        print(result)
        # Output:
        # {
        #     'language': 'Spanish',
        #     'translated_email': 'Hello, how are you? I hope everything is going well.',
        #     'summary': 'A friendly greeting and a wish for well-being.'
        # }
    """
    # Create Model
    llm = ChatOpenAI()
    
    # CREATE A CHAIN THAT DOES THE FOLLOWING:
    
    # Detect Language
    template1 = "Return the language this email is written in:\n{email}.\nONLY return the language it was written in."
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain_1 = LLMChain(llm=llm,
                     prompt=prompt1,
                     output_key="language")
    
    # Translate from detected language to English
    template2 = "Translate this email from {language} to English. Here is the email:\n"+email
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain_2 = LLMChain(llm=llm,
                     prompt=prompt2,
                     output_key="translated_email")
    
    # Return English Summary AND the Translated Email
    template3 = "Create a short summary of this email:\n{translated_email}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain_3 = LLMChain(llm=llm,
                     prompt=prompt3,
                     output_key="summary")
    
    seq_chain = SequentialChain(chains=[chain_1,chain_2,chain_3],
                            input_variables=['email'],
                            output_variables=['language','translated_email','summary'],
                            verbose=True)
    return seq_chain(email)

result = translate_and_summarize(spanish_email)
print(result['summary'])
