import json
import os
import re
import logging
from urllib.request import urlretrieve
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union
from tqdm import tqdm
import openai
import tiktoken
import asyncio

class recipe_manager_ai:
    """
    This class is used to manage the recipe manager AI.
    """
    
    def read_credential():
        with open('secrets.json', 'r') as f:
            credentials = json.load(f)
        return credentials
    
    def read_configs():
        # Load credentials from configs.json file
        with open('configs.json', 'r') as f:
            configs = json.load(f)
        return configs

    def __init__(self):
        
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s: %(message)s",
            datefmt="%b/%d %H:%M:%S",
            level=logging.INFO
        )

        # get chat completion parameters in configs.json file
        self.configs = recipe_manager_ai.read_configs()

        # Define the order in which prompts will be loaded
        self.prompt_load_order = [
            'prompt_role',
            'prompt_environment',
            'prompt_input_output_format',
             'prompt_query'
        ]

        self.prompt_image_load_order = [
            'prompt_image'
        ]

        self.AVAILABLE_MODELS = [  
            "gpt-4",
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-davinci-002",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "gpt2",
        ]
  
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing recipe_manager_ai class.")
        
        # Set OpenAI API key
        credentials = recipe_manager_ai.read_credential()
        openai.api_key = credentials["recipe_manager_ai"]["openai_api_key"]
        
        # Initialize list of ingredients
        self.ingredient_list = []
        
        # Set credentials
        self.credentials = credentials
        
        # Initialize messages list
        self.messages = []
        
        general_configs = self.configs['configs']['general']
        self.save_prompt_on_completion: bool = general_configs["save_prompt_on_completion"]
        self.markdown: bool = general_configs["markdown"]
        self.verbose: bool = general_configs["verbose"]
        self.isFakeAI: bool = general_configs["is_fake_ai"]
        
        # Set tiktoken encoding
        self.enc = tiktoken.get_encoding(general_configs["encoding_name"])
        
        chat_completion_configs = self.configs['configs']['recipe_manager_ai']['chat_completion']
        # Set maximum token length
        self.chat_completion_max_token_length = chat_completion_configs['max_token_length']
        
        # Set maximum completion length
        self.chat_completion_max_completion_length = chat_completion_configs["max_completion_length"]
        
        # Set temperature for text generation
        self.chat_completion_temperature = chat_completion_configs["temperature"]
        
        # Set number of completions to generate
        self.chat_completion_n= chat_completion_configs["n"]
        
        # Set top p value for text generation
        self.chat_completion_top_p = chat_completion_configs["top_p"]
        
        # Set frequency penalty for text generation
        self.chat_completion_frequency_penalty = chat_completion_configs["frequency_penalty"]
        
        # Set presence penalty for text generation
        self.chat_completion_presence_penalty = chat_completion_configs["presence_penalty"]
        
        # Set stop token for text generation
        self.chat_completion_stop = chat_completion_configs["stop"]
        
        # Set streaming mode to False
        self.chat_completion_stream = chat_completion_configs["stream"]
        
        # Set number of best completions to return
        self.chat_completion_best_of = chat_completion_configs["best_of"]
        
        # Set logprobs to 0
        self.chat_completion_logprobs = chat_completion_configs["logprobs"]
        
        # Set echo mode to False
        self.chat_completion_echo = chat_completion_configs["echo"]
        
        # Set model to use for text generation
        self.chat_completion_model = chat_completion_configs["model"]

        # Set prompt path for text generation
        self.chat_completion_output_path = chat_completion_configs["output_path"]
        self.chat_completion_prompt_path = chat_completion_configs["prompt_path"]

        image_generation_configs = self.configs['configs']['recipe_manager_ai']['image_generation']
        self.image_generation_n = image_generation_configs["n"]
        self.image_generation_size = image_generation_configs["size"] 
        self.image_generation_output_path = image_generation_configs["output_path"]
        self.image_generation_prompt_path = image_generation_configs["prompt_path"]

        db_configs = self.configs['configs']['recipe_manager_ai']['db']
        self.db_path = db_configs["path"]

    def main(self):
        #"""
        #   Main function for the recipe_manager_ai class
        #  :return: None
        # """
        self.logger.info("Main recipe manager AI")

        validate_model(self, self.chat_completion_model)
        
        # Ask the user if they want to delete the list of ingredients from memory
        recipe_response = input("Do you want to delete the list of ingredients from memory? Enter 1 if you want to delete, otherwise leave it blank.: ")
        
        # If the user wants to delete the list of ingredients, call the delete_ingredients_to_local_memory function
        if recipe_response == "1":
            delete_ingredients_to_local_memory(self)
        while True:
            try:
                # Define valid responses
                valid_responses = {'': 'add_item', '1': 'add_item', '2': 'submit_recipe', '3': 'remove_item', '4': 'remove_all'}

                # Ask the user if they want to continue or submit the recipe
                recipe_response = input("Do you want to continue? (1: Yes, 2: Submit Recipe, 3: Remove an item from the list, 4: Remove all items from the list): ")

                # Validate response input
                if recipe_response not in ['', '1', '2', '3', '4']:
                    raise Exception("Invalid input. Please enter '' or '1, 2, 3, 4'.: ")
                instructions = ""
                # Process response
                if recipe_response in valid_responses:
                    self.logger.info("Processing response: %s", recipe_response)

                    action = valid_responses[recipe_response]
                    if action == 'add_item':
                        self.logger.info("Adding item to the list.")
                        # Ask the user to enter the name, quantity, and unit of measurement for the item
                        new_ingredient = input("Enter the name, quantity, and unit of measurement for the item (e.g. apples-2-pounds): ")
                        
                        # Validate the format of the input string
                        if verify_format(self, new_ingredient):
                            
                            self.logger.info("Creating JSON item from input string.")
                            # Create a JSON item from the input string
                            json_ingredient = create_ingredient_json(self, new_ingredient)
                            # Check if the item is already in the list
                            if not has_ingredient(self, json_ingredient):
                                self.logger.info("Saving item to local memory.")
                                # Save the JSON item in local memory
                                save_ingredient_to_local_memory(self, json_ingredient)
                            else:
                                raise Exception("Item already in list")
                        else:
                            raise Exception("Invalid item format")        
                        continue
                    elif action == 'remove_item':
                        self.logger.info("Removing item from the list.")

                        item_name = input("Enter the name of the item you want to remove: ")
                        delete_ingredient_to_local_memory(self, item_name)
                        continue
                    elif action == 'remove_all':
                        self.logger.info("Removing all items from the list.")

                        delete_ingredients_to_local_memory(self)
                        continue
                    elif action == 'submit_recipe':
                        self.logger.info("Submitting recipe.")

                        instructions = input("Please provide the necessary instruction for the recipe, such as the type of dish, the region, allergies, cooking time, number of servings, and any ingredients to avoid. For example: type of dessert, Italian cuisine, gluten-free, nut allergy, no cinnamon, 6 servings, preparation time of no more than 60 minutes, microwave cooking. Leave blank if there are no instructions.: ")
                
                        is_strict_ingredients = input("Should the ingredients be strict? Please enter 'yes' or 'no': ").lower()
                        while is_strict_ingredients not in ['yes', 'no']:
                            if not is_strict_ingredients == 'yes' or not is_strict_ingredients == 'no':
                                raise Exception("Invalid input. Please enter 'yes' or 'no'.: ")
                    
                        self.logger.info("Creating recipe prompt.")
                        # Create a recipe prompt using the input from the request question and the ingridient in ingredient list
                        recipe_prompt_message = create_recipe_prompt(self, get_ingredient_list(self), instructions, is_strict_ingredients)
                        # Creating recipe from AI.
                        for i, recipe_prompt_message_modified in enumerate(tqdm(recipe_prompt_message, desc="Generating prompted..."), start=1):
                            self.logger.info("Genearate Prompted Text %d", i)

                        self.logger.info("Saving request to database.")
                        # Save the request to the database
                        save_request_to_db(self, recipe_prompt_message)

                        self.logger.info("Create recipe from AI. please wait...")
                        # Create a recipe from the AI
                        recipe_response = create_recipe_from_ai(self, recipe_prompt_message)

                        self.logger.info("Recipe from AI completed.")
                        image_url = ""
                        # loop through the response choices
                        for choice in recipe_response["choices"]:
                            self.logger.info("Response choice: %s", choice)
                            # get the content of the response choice message
                            contents = choice.message["content"].strip() 
                            
                            # remove the \n and spaces from the response
                            contents = contents.replace('\n', '')
                            contents = contents.replace('    ', '')

                            self.logger.info("Saving response to database.")
                            # Save the response to the database
                            save_response_to_db(self, recipe_response)
                            self.logger.info("Saving AI response to file.")
                            ts = get_timestamp(self)
                            save_generated_texts_to_file(self, recipe_response, ts)
                            try:
                                self.logger.info("Loading json prompt recipe content.")
                                #try to load json content IA response
                                json_data = json.loads(contents)
                                self.logger.info("json data: %s", json_data)

                                # Save format response to file
                                save_generated_texts_to_file(self, json_data, ts, "_JSON_")
                                
                                # Create an image prompt using the recipe
                                image_prompt = create_image_prompt(self, json_data)

                                # Generate the recipe image
                                filename, image_url, recipe_image_response = create_recipe_image_from_ai(self, image_prompt, ts)

                                self.logger.info("Image URL: %s", filename)
                                
                            except Exception as e:
                                self.logger.info("Error: %s", e)
                                continue
                        
                        self.logger.info("End of recipe generation.")
                        
                        return recipe_response, image_url, recipe_image_response
                    else:
                        raise Exception("Invalid response. Please leave it blank or '1', '2', '3', or '4'.") 
            except Exception as e:
                self.logger.info("Error: %s", e)
                continue

def validate_model(self, model: str) -> None:
    """
    Validate the model.
        
        Args:
            self (object): The object.
            model (str): The model to validate.
            
            Raises:
                ValueError: If the model is invalid.
            
            Returns:
                None
                    
    """

    if model not in self.AVAILABLE_MODELS:
        self.logger.info(f"Invalid model '{model}', available models: {', '.join(self.AVAILABLE_MODELS)}")
        raise ValueError(
            f"Invalid model '{model}', available models: {', '.join(self.AVAILABLE_MODELS)}"
        )

def has_ingredient(self, ingredient : str or dict) -> bool:
    """
    Check if an ingredient is already in the ingredient list.
    
    Args:
        self (object): The object.
        ingredient (str or dict): The ingredient to check. If provided as a string, it will be parsed
                                  as JSON. If provided as a dictionary, it will be used as-is.
    
    Returns:
        bool: True if the ingredient is already in the list, False otherwise.
    """
    try:
        self.logger.info("Checking if the item is already in the list.")
        # Check if the item is already in the list
        if isinstance(ingredient, str):
            self.logger.info("Parsing item from string.")
            _item = json.loads(ingredient)
        elif isinstance(ingredient, dict):
            self.logger.info("Parsing item from dictionary.")
            _item = ingredient
        else:
            self.logger.info("Invalid item format.")
            return False
        self.logger.info("Getting ingredient list.")
        ingredient_list = get_ingredient_list(self)
        for tmp_item in ingredient_list:
            self.logger.info("Parsing item from string.")
            tmp_item = json.loads(tmp_item)
            self.logger.info("Checking if the item is already in the list.")
            if _item["name"] == tmp_item["name"]:
                self.logger.info("Item already in list.")
                return True
        self.logger.info("Item not in list.")
        return False
    except:
        self.logger.info("Error checking if the item is already in the list.")
        return False

def verify_format(self, item : str) -> bool:
    """
    Verify if the item format is valid.
    
    Args:
        self (object): The object.
        item (str): The item to be verified.
        
    Returns:
        bool: True if the item format is valid, False otherwise.
    """
    try:
        self.logger.info("Verifying item format.")
        # Verify if the item format is valid
        item_info = item.split("-")
        if len(item_info) == 3:
            self.logger.info("Valid format.")
            try:
                float(item_info[1])
                self.logger.info("Valid format.")
                return True
            except:
                self.logger.info("Invalid format.")
                return False
        else:
            self.logger.info("Invalid format.")
            return False
    except:
        self.logger.info("Error verifying format.")
        return False
    
def create_ingredient_json(self, item  : str) -> str:
    """
    Create a JSON item from the input string.

    Args:
        self (object): The object.
        item (str): The item to be converted to JSON.

    Returns:
        str: The JSON item.
    """
    try:
    
        # Create a JSON item from the input string
        self.logger.info("Creating JSON item.")
        item_info = item.split("-")
        if len(item_info) >= 3:
            self.logger.info("Valid format.")
            item_dict = {"name": item_info[0], "quantity": item_info[1], "unit_of_measure": item_info[2]}
        else:
            # Handle the error here, for example by setting item_dict to None or raising an exception
            raise Exception("Invalid item format")
        
        # Convert the item dictionary to a JSON string
        self.logger.info("Converting item dictionary to JSON string.")
        json_item = json.dumps(item_dict)
        return json_item
    except:
        self.logger.info("Error creating JSON item.")
        return None
    
def save_ingredient_to_local_memory(self, json_item : str) -> bool:
    """
    Save an ingredient to local memory.

    Args:
        self (object): The object.
        json_item (str): The ingredient to be saved to local memory.

    Returns:
        None
    """
    try:
        # Load the list of ingredients from local memory
        ingredient_list = get_ingredient_list(self)

        # Append the new ingredient to the list
        ingredient_list.append(json_item)

        # Save the updated list back to local memory
        with open(self.db_path + "/items.json", "w") as f:
            json.dump(ingredient_list, f)
            return True
    except:
        self.logger.info("Error saving ingredient to local memory.")
        return False

def delete_ingredients_to_local_memory(self)-> bool:
    """
    Delete all ingredients in local memory.

    Args:
        self (object): The object.

    Returns:
        None
    """
    self.logger.info("Deleting all ingredients in local memory.")
    # Delete all items from the list in local memory
    try:
        
        with open(self.db_path + "/items.json", "w") as f:
            json.dump([], f)
            return True
    except:
        self.logger.info("Error deleting all ingredients in local memory.")
        return False

def delete_ingredient_to_local_memory(self, item_name)-> bool:
    """
    Delete an ingredient in local memory.

    Args:
        self (object): The object.
        item_name (str): The ingredient to be deleted in local memory.

    Returns:
        None
    """
    try:
        # Load the list of items from local memory
        ingredient_list = get_ingredient_list(self)

        # Remove the item with the specified name from the list
        for item in ingredient_list:
            item = json.loads(item)
            if item["name"] == item_name:
                ingredient_list.remove(item)

        # load the list of items from local memory
        with open(self.db_path + "/items.json", "w") as f:
            # Save the updated list back to local memory
            json.dump(ingredient_list, f)
            return True
    except:
        self.logger.info("Error deleting ingredient in local memory.")
        return False

def create_recipe_prompt(self, ingredient_list : dict, instructions : str, is_strict_ingredients : bool) -> list:
    """
    Create a recipe prompt using the input from the request question and the ingridient in ingredient list.

    Args:
        self (object): The object.
        ingredient_list (list): The list of ingredients.
        instruction (str): The instruction for the recipe.
        is_strict_ingredients (bool): Whether to use the ingredients in the ingredient list or not.
        
    Returns:
        str: The recipe prompt.
    """
    
    try:

        # load the prompt files in the specified order
        prompt = prompt_loading(self)
        # add instruction and is_strict_ingredients to the prompt
        ingredients_prompt = f'{{"instruction":"{instructions}",'
        ingredients_prompt += f'"is_strict_ingredients":"{is_strict_ingredients}",'
        temp_ingredients = ""
        i = 0
        for item in ingredient_list:
            self.logger.info("Creating JSON item.")
            
            # Create a JSON item from the input string
            item = json.loads(item)
            if i != 0:
                temp_ingredients += ","
            temp_ingredients += f'{{"name":"{item["name"]}","quantity":"{item["quantity"]}","unit_of_measure":"{item["unit_of_measure"]}"}}'
            i+=1
        
        # add the ingredients to the prompt
        ingredients_prompt += f'"ingredients":[{temp_ingredients}]}}'

        # check if the ingredients are in the correct format
        if not is_json_valid(self, ingredients_prompt):
            print(ingredients_prompt)
            print("Invalid JSON format")
            return ""

        print(ingredients_prompt)
        # Replace the placeholder in the prompt with the ingredients
        prompt = prompt.replace(f"[ingredients_prompt]", f"[{ingredients_prompt}]")

        if check_if_prompt_is_too_long(self, prompt):
            # find how to truncate the prompt - next version
            return ""

        print(prompt)

        pattern = r"\[(system|user|assistant)\]\s*(.*)"
        current_message = {}
        messages = []
        for line in prompt.split("\n"):
            match = re.match(pattern, line)
            if match:
                if current_message:
                    messages.append(current_message)
                current_message = {"role": match.group(1), "content": match.group(2).strip()}
            elif current_message:
                current_message["content"] += " " + line.strip()

        if current_message:
            messages.append(current_message)

        return messages
    except:
        self.logger.info("Error creating recipe prompt.")
        return []

def prompt_loading(self)-> str:
    """
    Load the prompt files in the specified order.

    Args:
        self (object): The object.

    Returns:
        None
    """
    self.logger.info("Loading the prompt files in the specified order.")
    prompt = ""
    
    try:
        
        for prompt_name in self.prompt_load_order:
            self.logger.info(f"Loading prompt file: {prompt_name}")
            fp_prompt = os.path.join(self.chat_completion_prompt_path, prompt_name + '.txt')
            with open(fp_prompt) as f:
                self.logger.info(f"Adding prompt file: {prompt_name}")
                prompt += f"{f.read()}"
                prompt += "\n\n"
        return prompt
    except:
        self.logger.info("Error loading prompt files.")
        return ""

def prompt_image_loading(self)-> str:
    """
    Load the prompt files in the specified order.

    Args:
        self (object): The object.

    Returns:
        None
    """
    try:
        self.logger.info("Loading the prompt files in the specified order.")
        prompt = ""
        
        for prompt_name in self.prompt_image_load_order:
            self.logger.info(f"Loading prompt image file: {prompt_name}")
            fp_prompt = os.path.join(self.image_generation_prompt_path, prompt_name + '.txt')
            with open(fp_prompt) as f:
                self.logger.info(f"Adding prompt file: {prompt_name}")
                prompt += f"{f.read()}"
                prompt += "\n\n"
        return prompt
    except:
        self.logger.info("Error loading prompt files.")
        return ""

def check_if_prompt_is_too_long(self, prompt: str)-> bool:
    """
    Truncate the prompt if it's too long.

    Args:
        self (object): The object.
        prompt (str): The prompt to be checked.

    Returns:
        None
    """
    #Encodes a string into tokens.
    tokens = self.enc.encode(prompt)
    if len(tokens) > self.chat_completion_max_token_length - \
                self.chat_completion_max_completion_length:
            print('Prompt too long. truncated.')
            # truncate the prompt by removing the oldest two messages
            self.messages = self.messages[2:]
            return True
    return False

def create_image_prompt(self, recipe_prompt: dict) -> str:
    """
    Create an image prompt using the input from the recipe prompt.

    Args:
        self (object): The object.
        recipe_prompt (str): The recipe prompt to be used in the image prompt.

    Returns:
        str: The image prompt.
    """ 

    # get the recipe name and ingredients from the recipe prompt
    recipe_name = recipe_prompt['recipe_name']
    ingredients = recipe_prompt['ingredients']

    self.logger.info(f"Recipe Name: {recipe_name}")
    ingredients_list = ""
    for ingredient in ingredients:
        ingredients_list = f"{ingredient['name']}: {ingredient['quantity']} {ingredient['unit_of_measure']}"
    
    self.logger.info(f"Ingredients: {ingredients_list}")
    self.logger.info("Creating an image prompt.")
    image_prompt = prompt_image_loading(self)
    image_prompt += f"\n\nRecipe Name:\n{recipe_name}\nIngredients:\n{ingredients_list}"
    self.logger.info(f"Image Prompt: {image_prompt}")
    return image_prompt

def save_request_to_db(self, recipe_prompt : str) -> bool:
    """
    Create a JSON object for the request and save it to the database.
    
    Args:
        request_question (str): The question of the request.
        recipe_prompt (str): The recipe prompt of the request.

    Returns:
        dict: The request.
    """
    try:
        # Create a JSON object for the request and save it to the database
        ingredient_list = get_ingredient_list(self)
        request = {"recipe_prompt": recipe_prompt}
        with open(self.db_path + "/requests.json", "a") as f:
            f.write(json.dumps(request) + "\n")
        return True
    except:
        self.logger.info("Error saving request to database.")
        return False
    
async def dispatch_openai_requests(
    self,
    messages_list: list[list[dict[str,Any]]],
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    completion = openai.Completion()
    
    async_responses = [
        openai.ChatCompletion.acreate(
             model = self.chat_completion_model,
                    messages=x,
                    temperature=self.chat_completion_temperature,
                    max_tokens=self.chat_completion_max_completion_length,
                    top_p=self.chat_completion_top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses) 

def create_recipe_from_ai(self, request : str) -> dict:
     # Generate a recipe using the AI
    try:
        if self.isFakeAI:
            self.logger.info("Generate a recipe using the FakeAI.")
            fake_json = json.loads('{"content":{"recipe_name":"Vietnamese Beef Noodle Salad","dateTime_utc":"2021 - 09 - 15 T19: 45: 00 Z","preparation_time":25,"cooking_time":15,"total_cooking_time":40,"servings":4,"ingredients":[{"name":"filet de boeuf","quantity":"500","unit_of_measure":"g"},{"name":"vermicelle de riz","quantity":"400","unit_of_measure":"g"},{"name":"Farine","quantity":"500","unit_of_measure":"g"}],"prepSteps":"Cook the vermicelli noodles according to the package","role":"assistant"}}')
            return fake_json
        else:
            # Attente de la réponse de l'API tout en affichant une barre de progression
            response = openai.ChatCompletion.create(
                model = self.chat_completion_model,
                messages=request,
                temperature=self.chat_completion_temperature,
                max_tokens=self.chat_completion_max_completion_length,
                top_p=self.chat_completion_top_p,
            )
            self.logger.info("Recipe done.")
            return response
    except:
        self.logger.info("Error generating recipe.")
        return None

def save_generated_texts_to_file(self, prompt: str, ts: str, suffix="") -> str:
    """ 
    Save the generated texts to a file.

    Args:
        self (object): The object.
        prompt (str): The prompt used to generate the texts.

    Returns:
        None
    """
    try:
        self.logger.info("Saving the generated texts to a file.")
        
        self.logger.info("Saving the generated texts to a file.")
        out_path = Path(self.chat_completion_output_path)
        
        self.logger.info(f"Saving output to {out_path}")
        out_path.mkdir(parents=True, exist_ok=True)
        filename = ""
        if self.verbose or not self.chat_completion_output_path:
            print(f"Prompt:\n{prompt}\n\n")
            self.logger.info(f"Saving output to {out_path}")
        if self.chat_completion_output_path:
            output_content = (
                f"{prompt}"
                if self.save_prompt_on_completion
                else prompt
            )  # add prompt to output if save_prompt is True
            filename = out_path / f"result_{suffix}{ts}.txt"
            output_file = filename
            output_file = (
                output_file.with_suffix(".md") if self.markdown else output_file
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_content)
        return filename
    except:
        self.logger.info("Error saving generated texts to a file.")
        return "no file created"

def get_timestamp(self):
    """
    Get the timestamp.

    Args:
        self (object): The object.

    Returns:
        str: The timestamp. format: YYYY-MM-DD_HHMMSS
    """

    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def create_recipe_image_from_ai(self, image_prompt : str, ts : str) -> tuple:
    """
    Generate a recipe image using the AI.

    Args:
        self (object): The object.
        image_prompt (str): The image prompt of the request.
        ts (str): The timestamp of the request.

    Returns:
        str: filename
        str: image_url
        str: recipe_image_response
    """
    image_url = ""
    try:
        self.logger.info("Generating a recipe image using the AI.")
        request_url = ""
        recipe_image_response = None
        if self.isFakeAI:
            #is fake AI
            filename = f'{self.image_generation_output_path}recipe_{ts}.png'
            image_url = 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-9ecno2hg2XOJdmvPbRbXQRLC/user-bfmLFpv0uW6To87XhVgdMXvc/img-fTlJXSocU0ErRaZJUADEw3CP.png?st=2023-05-12T02%3A39%3A14Z&se=2023-05-12T04%3A39%3A14Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-05-12T01%3A45%3A52Z&ske=2023-05-13T01%3A45%3A52Z&sks=b&skv=2021-08-06&sig=U58aKHFcS2reR%2BP0s%2Be5zn90wqGjJrfCmkZiplhHaAE%3D'
            recipe_image_response = urlretrieve(image_url, filename)
        else:
            self.logger.info("Generating a recipe image using the AI.")
            #add loading
            response = openai.Image.create(
                prompt=image_prompt,
                n=self.image_generation_n,
                size=self.image_generation_size
            )
            # Get the image URL
            image_url = response['data'][0]['url']
        self.logger.info("Image URL: " + image_url)
        
        self.logger.info("Downloading the image...")
        # Set the filename from the directory output and timestamp
        filename = f'{self.image_generation_output_path}image_recipe_{ts}.png'
        # Download the image
        recipe_image_response = urlretrieve(image_url, filename)
        return (filename, image_url, recipe_image_response)
    except Exception as e:
        self.logger.info("Error generating image: " + str(e))
        return ("", "", "")

def is_json_valid(self, json_string : str) -> bool:
    """
    Check if a JSON string is valid.

    Args:
        self (object): The object.
        json_string (str): The JSON string to be checked.

    Returns:
        bool: True if the JSON string is valid, False otherwise.
    """

    # Check if the JSON string is valid
    try:
        self.logger.info("Checking if the JSON string is valid...")
        json.loads(json_string)
        self.logger.info("The JSON string is valid.")
        return True
    except:
        self.logger.info("The JSON string is not valid.")
        return False

def save_response_to_db(self, response : str)-> bool:
    """
    Save a response to the database.

    Args:
        self (object): The object.
        response (dict): The response to be saved.

    Returns:
        None
    """
    try:   
        # Save the response to the database
        self.logger.info("Saving the response to the database...")
        with open(self.db_path + "/responses.json", "a") as f:
            self.logger.info("Successfully saved the response to the database.")
            f.write(json.dumps(response) + "\n")
        return True
    except:
        self.logger.info("Error saving the response to the database.")
        return False

def get_ingredient_list(self) -> list:
    """
    Get the list of ingredients from local memory.

    Args:
        self (object): The object.

    Returns:    
        list: The list of ingredients.
    """
    self.logger.info("Getting the list of ingredients from local memory...")
    # Read the items from local memory
    try:
        with open(self.db_path + "/items.json", "r") as f:
            self.logger.info("Successfully read the list of ingredients from local memory.")
            ingredient_list = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist yet, return an empty list
        self.logger.info("The list of ingredients doesn't exist yet.")
        ingredient_list = []
    return ingredient_list

app = recipe_manager_ai()
# Run the application
recipe_response, image_url, recipe_image_response = app.main()

#create the only the text for readme

