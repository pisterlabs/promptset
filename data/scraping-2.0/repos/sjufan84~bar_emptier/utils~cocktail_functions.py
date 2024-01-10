# Importing the necessary libraries
import os
import openai
import streamlit as st
import requests
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from typing import List, Optional, Union
import pandas as pd
from redis import Redis as RedisStore
import uuid





from dotenv import load_dotenv
load_dotenv()

# Get the OpenAI API key and org key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
elif 'total_inventory_ingredients_cost' not in st.session_state:
    st.session_state.total_inventory_ingredients_cost = []
elif 'total_ni_cost' not in st.session_state:
    st.session_state.total_ni_cost = 0.00,
elif 'total_cocktail_cost' not in st.session_state:
    st.session_state.total_cocktail_cost = 0.00
elif 'ni_ingredients' not in st.session_state:  
    st.session_state.ni_ingredients = []
elif 'total_inv_cost' not in st.session_state:
    st.session_state.total_inv_cost = 0.00
elif 'total_cocktail_cost' not in st.session_state:
    st.session_state.total_cocktail_cost = 0.00
elif 'total_drinks' not in st.session_state:
    st.session_state.total_drinks = 0

# Initialize a connection to the redis store
redis_store = RedisStore()

class CocktailIngredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    amount: float = Field(description="Amount of the ingredient")
    unit: str = Field(description="Unit of the ingredient")

# We want to create a parser object to parse the recipe into the variables we want using Pydantic
class CocktailRecipe(BaseModel):
    name: str = Field(description="Name of the cocktail recipe")
    ingredient_names: List[str] = Field(description="A list of the names of the ingredients in the cocktail.")
    ingredient_amounts: List[Union[float, int]]  = Field(description="A list of the amounts of the ingredients in the cocktail as floats.")
    ingredient_units: List[str] = Field(description="A list of the units of the ingredients in the cocktail.  Could be oz, splash,\
                                                    dash, etc.")
    ingredients_list: List[str] = Field(description="A list of the ingredients in the cocktail")
    instructions: List[str] = Field(description="Instructions for preparing the cocktail")
    garnish: str = Field(description="Garnish for the cocktail")
    glass: str = Field(description="Glass to serve the cocktail in")
    flavor_profile: Optional[str] = Field(description="Flavor profile of the cocktail")

    # Define a function to return the ingredients in a list of tuples
    @property
    def ingredients(self):
        return list(zip(self.ingredient_names, self.ingredient_amounts, self.ingredient_units))
    
    # Define a function to return the recipe text
    @property
    def recipe_text(self):
        return f"Recipe for {self.name}:\n\nIngredients:\n{self.ingredients}\n\nInstructions:\n{self.instructions}\n\nGarnish: {self.garnish}\n\nGlass: {self.glass}\n\nFlavor Profile: {self.flavor_profile}"
    

# Define a function to return a session id if the user does not have one,
# or to return the session id if the user already has one
def get_session_id():
    session_id = str(uuid.uuid4())
    return session_id

class NICost(BaseModel):
    total_ni_cost: float = Field(description="Total cost of non-inventory ingredients in the cocktail")

cost_parser = PydanticOutputParser(pydantic_object=NICost)
parser = PydanticOutputParser(pydantic_object=CocktailRecipe)

class RecipeService:
    def __init__(self, recipe: Optional[CocktailRecipe] = None):
        # Check to see if the user already has a recipe
        if recipe is None:
            # If the user does not have a recipe, create a new one
            self.recipe = None
        else:
            # If the user already has a recipe, load it
            self.recipe = recipe

       

    # A function to estimate the cost of non-inventory items in a cocktail by calling gpt-3.5
    def estimate_cost_of_non_inventory_items(self, ingredients: list) -> float:
        
        messages = [
        {
            "role": "system", "content" : f"You are a bar manager helping the user estimate the cost of ingredients {ingredients} in a cocktail you created for them.\
                Each ingredient in the list has the following format: (ingredient name, amount, unit).  Do your best to estimate the total cost of the ingredients\
                in the cocktail as a float in the following format: {cost_parser.get_format_instructions()}.  This is only an estimate, so you do not need to be exact.\
                Just approximate the cost and return it as a float."
        },
        {   
            "role": "user", "content": f"Given the ingredients and their amounts in {ingredients}, can you help me estimate the total cost of the ingredients?  It's okay\
                if you don't know the exact cost.  Just give me your best guess as a float in the following format: {cost_parser.get_format_instructions()}"
        },
        ]

    

        # Call the OpenAI API
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=messages,
            max_tokens=1000,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            temperature=1,
            top_p=0.9,
            n=1,
            )

            response = response.choices[0].message.content
            parsed_cost = cost_parser.parse(response)
            st.session_state.total_ni_cost = parsed_cost.total_ni_cost
            
            return parsed_cost.total_ni_cost
            

        except (requests.exceptions.RequestException, openai.error.APIError):
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                max_tokens=1000,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                top_p=0.9,
                n=1,
                )

                response = response.choices[0].message.content
                parsed_cost = cost_parser.parse(response)
                st.session_state.total_ni_cost = parsed_cost.total_ni_cost.values[0]




            except (requests.exceptions.RequestException, openai.error.APIError):
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0301",
                messages=messages,
                max_tokens=1000,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                temperature=1,
                top_p=0.9,
                n=1,
            )

                response = response.choices[0].message.content
                parsed_cost = cost_parser.parse(response)
                st.session_state.total_ni_cost = parsed_cost.total_ni_cost

        return parsed_cost.total_ni_cost           

    
    # Define a function to check the recipe for completeness
    def check_recipe_completeness(self, recipe):
        # Check to make sure the recipe has a name
        if recipe.name is None:
            return False
        if recipe.ingredients_list is None:
            return False
        elif recipe.ingredient_amounts == []:
            return False
        elif recipe.instructions == []:
            return False
        # If all of the above checks pass, return True
        else:
            return True

        
    
    def cost_recipe(self):
        total_cost = 0
        # This will take in a list of ingredients and return the cost of the ingredients per oz * the amount in the recipe
        # Load in the the inventory data
        inventory_service = st.session_state.inventory_service
        recipe = self.recipe
        # Convert the inventory data dictionary to a dataframe
        inventory = pd.DataFrame.from_dict(inventory_service.inventory, orient="columns")
        # Filter the inventory dataframe to only include the ingredients in the recipe
        for ingredient in recipe.ingredients:
            ingredient = list(ingredient)
            if ingredient[0] in inventory['Name'].values:
                # Get the Cost per oz of the ingredient
                cost_per_oz = inventory.loc[inventory['Name'] == ingredient[0]]['Cost per oz'].values[0]
                # Get the amount of the ingredient in the recipe
                amount = ingredient[1]
                # Get the cost of the ingredient in the recipe
                cost = float(cost_per_oz) * float(amount)
                # Append the cost to the list of costs
                ingredient.append(cost)
                st.session_state.total_inventory_ingredients_cost.append(ingredient)
                total_cost += cost
            else:
                st.session_state.ni_ingredients.append(ingredient)
        # Get the estimated cost of the non-inventory ingredients
        ni_cost = self.estimate_cost_of_non_inventory_items(st.session_state.ni_ingredients)
        st.session_state.total_inv_cost = total_cost
        st.session_state.total_ni_cost = ni_cost
        st.session_state.total_cocktail_cost = st.session_state.total_inv_cost + (st.session_state.total_ni_cost / 4)

        
        



    # Define the function to call the openai API
    def get_cocktail_recipe(self, liquor : str, theme : str, cuisine : str, cocktail_type : str, model_choice: str):
        
        # Define the first system message.  This let's the model know what type of output\
        # we are expecting and in what format it needs to be in.
        prompt = PromptTemplate(
            template = "You are a master mixologist helping a user use up the excess liquor {liquor}\
                        they have in their inventory by creating a creative and innovative cocktail recipe.\
                        featuring their excess liquor {liquor}.  This should be something that you would\
                        be proud to serve to a customer or friends.\
                        The recipe should be based around a theme {theme}, cuisine type {cuisine},\
                        and the type of cocktail {cocktail_type} the user wants to make.\
                        The recipe should include the ingredient names, the ingredient amounts,\
                             the ingredient units, the garnish, the glass, and a flavor profile. The cocktail should be returned in this format{format_instructions}.",
            input_variables = ["liquor", "theme", "cuisine", "cocktail_type"],
            partial_variables = {"format_instructions": parser.get_format_instructions()}
        )

        # Generate the system message prompt
        system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

        # Define the user message.  This is the message that will be passed to the model to generate the recipe.
        human_template = "Create a delcious cocktail recipe to help me use up my excess inventory."
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)   
        
        # Create the chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # format the messages to feed to the model
        messages = chat_prompt.format_prompt(liquor=liquor, theme=theme, cuisine=cuisine, cocktail_type=cocktail_type).to_messages()

        # Create a list of models to loop through in case one fails
        if model_choice == "gpt-3.5":
            models = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        else:
            models = ["gpt-4-0613", "gpt-4"]
        

        # Loop through the models and try to generate the recipe
        for model in models:
            chat = ChatOpenAI(model = model, temperature = 1, max_retries=3)

            try:
                recipe = chat(messages).content
                temp_parsed_recipe = parser.parse(recipe)
                if self.check_recipe_completeness(temp_parsed_recipe):
                    # Save the recipe to redis
                    self.recipe = temp_parsed_recipe
                    return temp_parsed_recipe
            except Exception as e:
                print(e)
                continue

        

                  
           
                    

                    
            

    # Create a function to generate a recipe based on the user's inventory if they have one uploaded
    def get_inventory_cocktail_recipe(self, inventory_list, liquor, cocktail_type, cuisine, theme, model_choice):
        messages = [
        {
             "role": "system", 
                "content" : f"You are a master mixologist helping the user create an innovative cocktail to use up their excess liquor, {liquor}.\
                            Return the recipe in the following format:\n{parser.get_format_instructions()}.\n" 
        },
        {   
            "role": "user", "content": f"Given the following parameters: the name of the liquor {liquor} I am trying to use up, the type of cocktail {cocktail_type}, the theme {theme},\
                                    and the style of cuisine {cuisine} to pair it with, please help me come up with a creative cocktail featuring {liquor} with a fun and creative name that doesn't necessarily include the name of the spirit or the theme.\
                                    Please prioritize using the ingredients I have on hand in {inventory_list}, but if it compromises the quality of the recipe, use a different ingredient.\
                                    Please be as specific as possible with your instructions."
        },
        {
            "role":"user", "content": f"Please use the following format:\n{parser.get_format_instructions()}\n."
        }
        ]
         # Based on the model choicce, Create a list of models to loop through in case one fails
        if model_choice == "gpt-3.5":
            models = ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        else:
            models = ["gpt-4-0613", "gpt-4"]
        # Loop through the models and try to generate the recipe
        for model in models:
            try:
                response = openai.ChatCompletion.create(
                    model = model,
                    messages = messages,
                    max_tokens = 750, 
                    temperature = 1,
                    top_p = 1,
                    frequency_penalty=0.5,
                    n=1
                )

                recipe = response.choices[0].message.content
                temp_parsed_recipe = parser.parse(recipe)
                if self.check_recipe_completeness(temp_parsed_recipe):
                    # Save the recipe to redis
                    self.recipe = temp_parsed_recipe
                    return temp_parsed_recipe
            except Exception as e:
                print(e)
                continue
                            
    # Define a function to return the total amount of drinks we can make based on the amount of the chosen spirit
    # we have in our inventory
    def get_total_drinks(self):
        recipe_service = st.session_state.recipe_service
        recipe = recipe_service.recipe
        inventory_service = st.session_state.inventory_service
        inventory = inventory_service.inventory
        # Get the amount of the chosen spirit in the recipe
        spirit_amount = recipe.ingredients[0][1]
        # Convert the spirit amount to ml from oz
        spirit_amount_ml = spirit_amount * 29.5735
        # Locate the total amount of the spirit in the inventory
        inventory_df = pd.DataFrame.from_dict(inventory, orient = 'columns')
        inventory_df = inventory_df[inventory_df['Name'].str.lower() == recipe.ingredients[0][0].lower()]
        # Find the values from the "Quantity" and "Volume per Unit" columns
        quantity = inventory_df['Quantity'].values[0]
        volume_per_unit = inventory_df['Volume per Unit (ml)'].values[0]
        total_volume = quantity * volume_per_unit
        # Calculate the total number of drinks we can make
        total_drinks = total_volume / spirit_amount_ml
        st.session_state.total_drinks = total_drinks

        return total_drinks


    
    # Create a function to display the cost of the recipe
    def display_cost(self):
        recipe_service = st.session_state.recipe_service
        recipe = recipe_service.recipe
        inventory_service = st.session_state.inventory_service
        inventory = inventory_service.inventory
        # Create two columns -- one two display the recipe text and the cost per recipe, the other to display the profit
        col1, col2 = st.columns(2, gap = 'medium')
        with col1:
            # Display the recipe name
            st.markdown(f'**Recipe Name:** {recipe.name}')
            # Display the recipe ingredients
            st.markdown('**Ingredients:**')
            # Check to see if the name of each ingredient is in the inventory dataframe regardless of case, and if it is, display it in red
            # If they are not in the inventory dataframe, display them in black
            for ingredient in st.session_state.total_ingredients_cost:
                # Display the ingredient 
                st.markdown(f'* :red[{ingredient[0]}]: ${float(ingredient[1]):.2f} {ingredient[2]} = ${float(ingredient[3]):.2f}')
            for ingredient in st.session_state.ni_ingredients:
                st.markdown(f'* {ingredient[0]}: ${float(ingredient[1]):.2f} {ingredient[2]} ')
            # Display the total cost of the recipe
            st.markdown(f'**Total Cost of inventory ingredients:** ${st.session_state.total_inv_cost:.2f}')
            st.markdown(f'**Total Cost of non-inventory ingredients:** ${st.session_state.total_ni_cost:.2f}')
            st.markdown(f'**Total Cost of recipe:** ${st.session_state.total_cocktail_cost:.2f}')

            
        with col2:
            # Calculate and display total costs and the potential profit
            st.markdown(f'**Total cost to use up the amount of {st.session_state.chosen_spirit} in your inventory:**')
            st.markdown(f'You can make **{st.session_state.total_drinks}** of the "{recipe.name}" with the amount of {st.session_state.chosen_spirit} you have in your inventory.')

            total_drinks_cost = st.session_state.total_cocktail_cost * st.session_state.total_drinks
            st.write(f'The total cost of the recipe for {st.session_state.total_drinks} drinks is ${total_drinks_cost:.2f}.')
            # Display the potential profit
            st.markdown('**Potential Profit:**')
            # Create a slider that allows the user to select the price of the drink they want to sell it for
            st.write('Select the price you want to sell the drink for:')
            price = st.slider('Price', min_value=10, max_value=20, value=10, step=1)

            # Calculate the profit
            total_profit = (st.session_state.total_drinks * price) - total_drinks_cost

            # Profit per drink
            profit_per_drink = price - st.session_state.total_cocktail_cost

            # Display the profit
            st.write(f'The total profit for {st.session_state.total_drinks} drinks is ${total_profit:.2f}.')
            st.write(f'The profit per drink is ${profit_per_drink:.2f} or {(profit_per_drink / price) * 100:.2f}%.')

        st.text("")
        st.text("")
        
        # Set the value of the chosen_spirit to the amount from the "Total Value" column in the inventory dataframe.  Match the name of the chosen_spirit to the name in the inventory dataframe regardless of case
        df = pd.DataFrame.from_dict(inventory, orient = 'records')
        total_value = df[df['Name'].str.lower() == st.session_state.chosen_spirit.lower()]['Total Value'].values[0]
        # Note the difference in the value of the chosen_spirit in inventory and the total profit
        st.success(f"Congratulations!  You turned \${total_value:.2f} worth of {st.session_state.chosen_spirit} into ${total_profit:.2f} worth of profit!")

        # Create a function to generate a training guide
        #Establish the function to submit the text of the recipe to the model and generate the training guide

    def generate_training_guide(self):
        # Set the recipe text to the recipe text from the recipe service
        recipe_text = self.recipe.recipe_text
        # Create the messages for the model
        messages = [
            {"role" : "system", "content" : f"Generate a detailed training guide for a cocktail recipe for a restaurant pre-shift staff education,\
            focusing on the history and specifics of the ingredients, the techniques used, and the flavor profile of the drink. The recipe is as\
            follows: {recipe_text}."
            },
        ]
        # Define the parameters for the API call
        params = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 750,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "temperature": 1,
            "top_p": 0.9,
            "n": 1,
        }
        
        # Call the OpenAI API and handle exceptions
        try:
            response = openai.ChatCompletion.create(**params)
        except (requests.exceptions.RequestException, openai.error.APIError):
            params["model"] = "gpt-3.5-turbo-0301"
            response = openai.ChatCompletion.create(**params)

        # Return the the response as training guide
        guide = response.choices[0].message.content

        return guide


