# Copyright (c) 2023, Geordie Everitt and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document
import os
from pprint import pprint
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


cotc_settings = frappe.get_single("Cult Settings")

os.environ["OPENAI_API_KEY"]=cotc_settings.openai_api_key
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=cotc_settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"]="cult-of-the-carrot"

class Ingredient(BaseModel):
    name: str = Field(..., description="The name of the ingredient")
    quantity: float = Field(..., description="The quantity of the ingredient as a float")
    unit: Optional[str] = Field(None, description="The unit of measurement (e.g. cup, tsp, etc.)")
    prep: Optional[str] = Field(None, description="Preparation instructions (e.g. chopped, sliced, etc.)")

class Instruction(BaseModel):
    instruction: str = Field(..., description="The action to be performed in this step")
    duration: int = Field(0, description="The duration of the step in seconds")

class RecipeModel(BaseModel):
    title: str = Field(..., description="The title of the recipe")
    summary: str = Field(..., description="A recipe-book style summary of the recipe")
    # cooking_time: int = Field(..., description="Total Cooking and Prep time in seconds")
    ingredients: List[Ingredient] = Field(..., description="List of ingredients")
    instructions: List[Instruction] = Field(..., description="List of instructions")
    # tags: List[str] = Field([], description="Tags associated with the recipe, like 'GBOMBS' or 'Daily Dozen'")
    # sage: str = Field(None, description="Associated Sage or guru")

class Recipe(Document):
	pass

@frappe.whitelist()
def create_recipe_from_form(prompt_value):
    logger.debug(f"Creating new recipe from prompt: {prompt_value}")
    parser = PydanticOutputParser(pydantic_object=RecipeModel)
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate(
        template="""You are a vegan chef designing tasty and nutritious recipes.
            No animal products are permitted in the recipes you generate, and they should use whole, unprocessed ingredients as much as possible.
            {format_instructions}
            {input_text}""",
        input_variables=["input_text"],
        partial_variables={"format_instructions": format_instructions}
    )
    messages = [[HumanMessage(content=prompt.format(input_text=prompt_value))]]
    model = ChatOpenAI(temperature=0.5)
    response = model.generate(messages)
    answer = response.generations[0][0].text
    recipe_model = parser.parse(answer)
    
    create_recipe_document(prompt_value, recipe_model, response)
    return f"Recipe created: {recipe_model.title}"

def create_or_get_ingredient(name):
    logger.debug(f"Creating or getting ingredient: {name}")
    existing_ingredient = frappe.db.exists("Ingredient", {"ingredient_name": name})
    if existing_ingredient:
        logger.debug(f"Ingredient already exists: {existing_ingredient}")
        return existing_ingredient
    new_ingredient = frappe.new_doc("Ingredient")
    new_ingredient.ingredient_name = name
    retval = new_ingredient.insert()
    frappe.db.commit()
    return retval.name

def create_recipe_document(prompt_value, recipe_model, llm_response):
    logger.info("Creating new recipe document")
    recipe_doc = frappe.new_doc("Recipe")
    recipe_doc.title = recipe_model.title
    recipe_doc.summary = recipe_model.summary
    recipe_doc.prompt = prompt_value
    recipe_doc.llm_response = llm_response

    for ingredient in recipe_model.ingredients:
        ingredient_link = create_or_get_ingredient(ingredient.name)
        recipe_doc.append("ingredients", {
            "ingredient": ingredient_link,
            "amount": ingredient.quantity,
            "unit": ingredient.unit,
            "prep": ingredient.prep
        })

    for instruction in recipe_model.instructions:
        recipe_doc.append("instructions", {
            "instruction": instruction.instruction,
            "duration": instruction.duration
        })

    try:
        recipe_doc.insert()
        frappe.db.commit()
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "Error in creating Recipe")
        frappe.db.rollback()
