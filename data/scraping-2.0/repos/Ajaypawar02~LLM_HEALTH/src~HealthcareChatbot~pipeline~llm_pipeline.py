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
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser,
)
from HealthcareChatbot.components.get_my_database import DataBase
from HealthcareChatbot.components.get_my_database import config_manager
from typing import List, Dict
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class Extraction(BaseModel):
    new_recipe_name: str = Field(description="new name of the recipe")
    ingredient_substitute: List = Field(
        description="List of ingredient substitute. This should not have any dict. It should be a python List")
    new_recipe: List = Field(
        description="List of new recipe based on the ingredients substituted.")


class Model:
    def __init__(self, db: DataBase):
        self.config_details_model = config_manager.get_model_config()
        self.llm = ChatOpenAI(temperature=self.config_details_model.temperature,
                              model_name=self.config_details_model.model_name)
        self.db = db.fetch_database()
        self.parser = PydanticOutputParser(pydantic_object=Extraction)
        self.prompt_template = """You are provided with the {recipe_name} ,{ingredients}, {ingredients_to_replace} and {ingredient_preparation} . You need to replace \
                                {ingredients_to_replace} with the ingredients present in the {ingredients_database} and suggest the new_recipe_name which should be relevant and must have its existence based on your knowledge.
                                Instructions
                                '''
                                1. Strictly Dont't used the replaced ingredients again. This is very strict information
                                2. Make the recipe based on all the ingredients substituted
                                '''
                                {format_instructions}
                                
                                """

        self.prompt = PromptTemplate(
            input_variables=["recipe_name", "ingredients", "ingredients_to_replace",
                             "ingredient_preparation", "ingredients_database"],
            template=self.prompt_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()},
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def database_list(self, query):
        incredients_database = []
        mycursor = self.db.cursor()
        mycursor.execute(query)
        recipe_rows = mycursor.fetchall()
        recipe_df = pd.DataFrame(recipe_rows, columns=[
            desc[0] for desc in mycursor.description])
        for i, data in recipe_df.iterrows():
            # print(data["name"])
            incredients_database.append(data["name"])
            # print("---------------------------------------------------")
        return incredients_database

    def run_chain(self, api_params: Dict):
        ans = self.chain.run({
            'recipe_name': api_params["recipe_name"],
            'ingredients': api_params["ingredients"],
            'ingredients_to_replace': api_params["ingredients_to_replace"],
            'ingredient_preparation': api_params["ingredient_preparation"],
            "ingredients_database": self.database_list("SELECT * FROM ingredients")
        })
        try:
            output = self.parser.parse(ans)
            # print(output)
        except:
            fix_parser = OutputFixingParser.from_llm(
                parser=self.parser, llm=self.llm)
            output = fix_parser.parse(ans)

        # print(ingredients)
        # print(output.new_recipe)
        return output
    
    async def arun_chain(self, api_params: Dict):
        ans = self.chain.arun({
            'recipe_name': api_params["recipe_name"],
            'ingredients': api_params["ingredients"],
            'ingredients_to_replace': api_params["ingredients_to_replace"],
            'ingredient_preparation': api_params["ingredient_preparation"],
            "ingredients_database": self.database_list("SELECT * FROM ingredients")
        })
        try:
            output = self.parser.parse(ans)
            # print(output)
        except:
            fix_parser = OutputFixingParser.from_llm(
                parser=self.parser, llm=self.llm)
            output = fix_parser.parse(ans)

        # print(ingredients)
        # print(output.new_recipe)
        return output


if __name__ == "__main__":
    import time
    start = time.time()
    db = DataBase()
    model = Model(db)
    api_params = {
        "recipe_name": "Keto Coconut Pancakes",
        "ingredients": [
            "egg",
            "coconut flour",
            "coconut_milk",
            "vanilla extract",
            "baking soda",
            "coconut oil"
        ],
        "ingredients_to_replace": [
            "coconut milk",
            "coconut flour"
        ],
        "ingredient_preparation": [
            "4 medium eggs, whisked",
            "0.5 cup of coconut flour(56g)",
            "1 cup of coconut milk(240 ml)",
            "2 cup of vanilla extract (10 ml)",
            "4 gm of backing soda",
            "60 ml coconut oil"
        ]
    }

    output = model.run_chain(api_params)
    print(output)
    end = time.time()
    print(end-start)
