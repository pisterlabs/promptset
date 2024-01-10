
import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from template import template_string

def wrapper(list):
    load_dotenv()
    list_str = f"{list}"
    prompt_template = ChatPromptTemplate.from_template(template_string)
    recipe_prompt = prompt_template.format_messages(list= list_str)

    # Call OpenAI 

    chat = ChatOpenAI(temperature=0.0)
    recipe_str= chat(recipe_prompt)
    output_str = recipe_str.content

    recipe_dict = json.loads(output_str)
    return recipe_dict
