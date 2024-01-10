from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import tiktoken
import config
import os


def num_tokens_from_string(string: str, encoding_name:str, model_name: str) -> int:
    """Returns the number of tokens used by string"""
    encoding = tiktoken.get_encoding(encoding_name)
    encoding = tiktoken.encoding_for_model(model_name)
    count_list= encoding.encode("tiktoken is great!")
    print (f"number of token are: {len(count_list)}")
    token_length= len(count_list)
    return token_length

def creative_writer(input: str, input_2: str):
    
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=config.secret_key["openai_key"])

    # Notice "plot" below, that is a placeholder for another value later
    #this type of prompting method is generally called few-shot prompting
    template = """ 
    I want you to act as an award winning novelist. 
    Your sole purpose is to help novelists and writers complete their engaging and creative script that can captivate its viewers based on their exisiting plot : {plot} and genre: {genre} of their choice. 
    Keep in mind the below when creating the story:
    - Start with coming up with interesting characters if necessary and the setting of the story
    - Create dialogues between the characters etc. 
    - Create an exciting story filled with twists and turns that keeps the viewers in suspense until the end. 
    - Please limit the response to 250 words.
    """

    prompt = PromptTemplate(
    input_variables=["plot","genre"],
    template=template,
    )

    final_prompt = prompt.format(plot=input,genre=input_2)
    #print (f"Final Prompt: {final_prompt}")
    #print ("-----------")
    #print (f"LLM Output: {llm(final_prompt)}")
    return llm(final_prompt)


