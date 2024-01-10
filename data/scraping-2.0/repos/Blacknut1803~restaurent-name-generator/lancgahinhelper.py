from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
import time 
import os 
# os.environ['OPENAI_API_KEY'] = "sk-"
llm = OpenAI(temperature=0.7)

def generate_restaurent_name_and_items(cuisine):
        
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine}. Suggest a fancy name for my restaurant."
    )
    name_chain = None
    try :
        name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    except Exception as e :
        if "Rate limit reached" in str(e):
             time.sleep(20)
        else :
            print(f"An error occurred: {e}")
        
    
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest me some menu items for {restaurant_name}."
    )
    food_items_chain = None
    try :
         food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    except Exception as e :
        if "Rate limit reached" in str(e):
             time.sleep(20)
        else :
            print(f"An error occurred: {e}")
    


    chain1 = SequentialChain(
        chains=[chain for chain in [name_chain, food_items_chain] if chain is not None],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )

    return chain1({'cuisine': cuisine})

