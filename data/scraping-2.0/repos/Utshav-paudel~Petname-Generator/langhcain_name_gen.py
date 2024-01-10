#@ Pet-name genrator using langchain and open ai
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os 
from dotenv import load_dotenv
load_dotenv()
# for running locally
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def pet_name_gen(animal_type,animal_color,OPENAI_API_KEY):
    llm = OpenAI(temperature=0.6, openai_api_key = OPENAI_API_KEY)                           # decide creativity level
    prompt_template = PromptTemplate(input_variables=['animal_type','animal_color'],
                                     template="I have {animal_type} with color {animal_color} Generate me 5 cool {animal_type} name ")
    pet_name = LLMChain(llm=llm, prompt=prompt_template)    # chaining llm with prompt template
    response = pet_name({'animal_type':animal_type, 'animal_color':animal_color}) # takes input_var
    return response

if __name__=="__main__":
    print(pet_name_gen("cat","white and brown"))