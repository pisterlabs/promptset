from langchain.llms import GooglePalm
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
load_dotenv()

api_key = os.getenv('google_api_key')

model = GooglePalm(google_api_key=api_key,temperature=0)

def get_restaurant_name_menu(cuisine):

    prompt_name = PromptTemplate(
        input_variables=['cuisine'],
        template="Can I get a name for {cuisine} retstaurant, return only one name"
    )
    name_chain = LLMChain(llm=model,prompt=prompt_name,output_key='cuisine_name')

    prompt_menu = PromptTemplate(
        input_variables=['cuisine_name'],
        template="Suggest some menu items for {cuisine_name}. Return it as a comma separated string"
    )
    menu_chain = LLMChain(llm=model, prompt=prompt_menu,output_key='menu_list')

    main_chain = SequentialChain(
        chains=[name_chain,menu_chain],
        input_variables=['cuisine'],
        output_variables=['cuisine_name','menu_list']
    )

    response = main_chain({"cuisine":cuisine})
    return response

if __name__ == "main":
    print(get_restaurant_name_menu({'cuisine':"Mexican"}))




