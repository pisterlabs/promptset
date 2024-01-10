# Load packages
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

# Declare functions
def SendPromptToChatGPT(prompt_template,
                        user_input,
                        system_message="You are a helpful assistant.",
                        # LLM parameters
                        openai_api_key=None,
                        temperature=0.0,
                        chat_model_name="gpt-3.5-turbo",
                        verbose=True):
    # If OpenAI API key is not provided, then try to load from .env file
    if openai_api_key is None:
        load_dotenv()
        try:
            openai_api_key = os.environ['OPENAI_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a OpenAI API key, visit https://platform.openai.com/")
    
    # Ensure that user_input is a dictionary
    if type(user_input) != dict:
        raise ValueError("user_input must be a dictionary with the variable in the prompt template as the key and text you want plugged into the template as the value.")
    
    # Ensure that each key in user_input is in the prompt template
    for key in user_input.keys():
        if "{" + key + "}" not in prompt_template:
            raise ValueError("The key '" + key + "' in user_input is not in the prompt template.")
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create an instance of the ChatGPT chat model
    chat_model = ChatOpenAI(openai_api_key=openai_api_key)
    
    # Create the ouput parser
    output_parser = StrOutputParser()
    
    # Create chain using lanchain's expression language (LCEL)
    chain = prompt | chat_model | output_parser

    # Send the system and user messages as a one-time prompt to the chat model
    response = chain.invoke(user_input)
    
    # Return the response
    return response

 
# # Test function
# response = SendPromptToChatGPT(
#     prompt_template="""
#     Break this key intelligence question into less than four sub-questions: {key_intelligence_question}
#     """,
#     user_input={
#         "key_intelligence_question": "What targets are Hamas most likely to strike next in Israel?"
#     },
#     system_message="""
#         You are a project manager. You specialize in taking a key intelligence question and breaking it down into sub-questions. 
#         When creating the sub-questions, identify the main components of the original question. What are the essential elements or variables that the decision maker is concerned about?
#     """,
#     openai_api_key=open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# )
