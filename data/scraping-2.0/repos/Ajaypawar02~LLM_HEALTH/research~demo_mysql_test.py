# import mysql.connector
# import pandas as pd
# from bs4 import BeautifulSoup
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.agents.agent_types import AgentType

# from langchain.agents import create_csv_agent
# import os
# os.environ["OPENAI_API_KEY"] = "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"

# # Establish a connection to the MySQL database
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="abstract-programmer",
#     password="example-password",
#     database="sumeet_health"
# )

# # Create a cursor object to execute SQL queries
# mycursor = mydb.cursor()

# # Execute the SQL query to fetch specific columns from the recipe table
# recipe_query = "SELECT * FROM recepie"
# mycursor.execute(recipe_query)
# recipe_rows = mycursor.fetchall()

# # print(recipe_rows)

# for desc in mycursor.description:
#     print(desc[0])

# def clean_html(html_text):
#     soup = BeautifulSoup(html_text, 'html.parser')
#     cleaned_text = ' '.join(soup.stripped_strings)
#     return cleaned_text

# recipe_df = pd.DataFrame(recipe_rows, columns=[desc[0] for desc in mycursor.description])
# recipe_df["cleaned_incredients"] = recipe_df["ingredients"].apply(lambda x : clean_html(x))
# # recipe_df.to_csv("data.csv", index = False)

# selected_columns = recipe_df[['id', 'name', 'cleaned_incredients']]

# # Print the resulting DataFrame
# print(selected_columns)
# selected_columns.to_csv("selected_columns.csv", index=False)


# agent = create_csv_agent(
#     OpenAI(temperature=0, model_name = "gpt-3.5-turbo-0613"),
#     "/home/ajay/Freelancing/LLM_HEALTH/selected_columns.csv",
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )
# agent.run("Can you suggest some recipe for breakfast and also mention its recipe ")


import mysql.connector
import pandas as pd
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from pydantic import BaseModel, Field
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import os
os.environ["OPENAI_API_KEY"] = "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"

# Establish a connection to the MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="abstract-programmer",
    password="example-password",
    database="sumeet_health"
)

# Create a cursor object to execute SQL queries
mycursor = mydb.cursor()

# Execute the SQL query to fetch specific columns from the recipe table
recipe_query = "SELECT * FROM ingredients"
mycursor.execute(recipe_query)
recipe_rows = mycursor.fetchall()

# print(recipe_rows)

for desc in mycursor.description:
    print(desc[0])

# clean the html tags from the string


def clean_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = ' '.join(soup.stripped_strings)
    return cleaned_text


recipe_df = pd.DataFrame(recipe_rows, columns=[
                         desc[0] for desc in mycursor.description])
# # recipe_df["cleaned_incredients"] = recipe_df["ingredients"].apply(lambda x : clean_html(x))
recipe_df.to_csv("data.csv", index=False)
incredients_database = []
for i, data in recipe_df.iterrows():
    print(data["name"])
    incredients_database.append(data["name"])
    print("---------------------------------------------------")

print(incredients_database)


class Extraction(BaseModel):
    new_recipe_name: str = Field(description="new name of the recipe")
    ingredient_substitute: str = Field(
        description="Just provide the ingredients_substitude\n")
    new_recipe: str = Field(
        description="new recipe based on the ingredients replaced example. Don't used replaced ingredients again this is strict instruction'''x (g/ml) of p \n y (g/ml) of q''' .")


llm = ChatOpenAI(temperature=0.0, model_name="gpt-4-0613")
prompt_template = """You are provided with the {recipe_name} ,{ingredients}, {ingredients_to_replace} and {ingredient_preparation} . You need to replace \
{ingredients_to_replace} with the ingredients present in the {ingredients_database} and suggest the new_recipe_name which should be relevant and must have its existence based on your knowledge.
Output
'''
new recipe name,
ingredient_substitute, 
new recipe based on the ingredient_substitute and remaining previous ingredients excluding the replaced ingredients
{format_instructions}
'''
"""

parser = PydanticOutputParser(pydantic_object=Extraction)
prompt = PromptTemplate(
    input_variables=["recipe_name", "ingredients", "ingredients_to_replace",
                     "ingredient_preparation", "ingredients_database"],
    template=prompt_template,
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)
chain = LLMChain(llm=llm, prompt=prompt)

ans = chain.run({
    'recipe_name': "Keto Coconut Pancakes",
    'ingredients': ['egg', 'coconut flour', 'coconut_milk', 'vanilla extract', 'baking soda', 'coconut oil'],
    'ingredients_to_replace': ["egg", "coconut flour"],
    'ingredient_preparation': ['4 medium eggs, whisked', '0.5 cup of coconut flour(56g)', '1 cup of coconut milk(240 ml)', '2 cup of vanilla extract (10 ml)', '4 gm of backing soda', '60 ml coconut oil'],
    "ingredients_database": incredients_database
    })
output = parser.parse(ans)
print(output)