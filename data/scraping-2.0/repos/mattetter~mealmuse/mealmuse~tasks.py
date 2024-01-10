import os
import re
import json
import openai
import time
from dotenv import load_dotenv

from mealmuse import db, celery
from mealmuse.models import User, Pantry, Item, ShoppingList, MealPlan, Recipe, Day, Meal, RecipeItem, ShoppingListItem, PantryItem, UserProfile, Equipment, Allergy, Diet, users_recipes, recipes_mealplans, recipes_meals  # import the models if they are used in the utility functions
from mealmuse.exceptions import InvalidOutputFormat  
from mealmuse.prompts  import recipes_prompt_35turbo_v1, meal_plan_system_prompt_gpt4_v2, pantry_items_prompt_gpt_4_v1

from test_data import get_recipe, meal_plan_output_gpt_4_v2

load_dotenv(".env")

openai.api_key = os.getenv("OPENAI_API_KEY")
RECIPES_TASK = recipes_prompt_35turbo_v1
RECIPE_MODEL = "gpt-3.5-turbo-16k"

MEAL_PLAN_TASK = meal_plan_system_prompt_gpt4_v2
MEAL_PLAN_MODEL = "gpt-4"

PANTRY_ITEMS_TASK = pantry_items_prompt_gpt_4_v1

def create_app_instance():
    from mealmuse import create_app  # Adjust this import to your actual function
    app = create_app('config.DevelopmentConfig') 
    return app


@celery.task
def generate_meal_plan(meal_plan_id, user_id, temp):

    # get meal plan from openai
    meal_plan_output = fetch_meal_plan_from_api(meal_plan_id, user_id, temp)

    # fake api call for testing
    # meal_plan_output = meal_plan_output_gpt_4_v2 

    # save generated meal plan with user selections to database
    meal_plan_id = save_meal_plan_output_with_context(meal_plan_output, meal_plan_id, user_id)

    # fetch recipe details in parallel
    fetch_recipe_details_with_context(meal_plan_id, user_id)

    return meal_plan_id


# Meal Plan generation; wrap the recipe api call in a function to be used in parallel
def fetch_recipe_details_with_context(meal_plan_id, user_id):
    app = create_app_instance()
    with app.app_context():
        try:
            meal_plan = MealPlan.query.filter_by(id=meal_plan_id).first()
            profile = UserProfile.query.filter_by(user_id=user_id).first()
            temp = profile.recipe_temperature
            # create a list of recipe ids in plan
            recipe_ids = []
            for recipe in meal_plan.recipes:
                recipe_ids.append(recipe.id)
            db.session.remove()

        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
        result = [fetch_recipe_details.delay(recipe_id, temp) for recipe_id in recipe_ids]
        return result


# Meal Plan generation: generate a single recipe
@celery.task
def swap_out_recipe(recipe_id, user_id):
    app = create_app_instance()
    with app.app_context():
        try:
            user = User.query.filter_by(id=user_id).first()
            old_recipe = Recipe.query.filter_by(id=recipe_id).first()
            meal = Meal.query.filter_by(id=old_recipe.meals[0].id).first()
            day = Day.query.filter_by(id=meal.day_id).first()
            meal_plan_id = day.meal_plan_id
            profile = UserProfile.query.filter_by(user_id=user_id).first()
            recipe_temperature = profile.recipe_temperature or 1
            meal_plan = MealPlan.query.filter_by(id=meal_plan_id).first()

            # get the recipe specific details
            recipe_cost = old_recipe.cost
            recipe_time = old_recipe.time
            recipe_serves = old_recipe.serves   
            recipe_cuisine = old_recipe.cuisine

            
            # disassociate the old recipe from the meal
            meal.recipes.remove(old_recipe)
            # disassociate the old recipe from the meal plan
            meal_plan.recipes.remove(old_recipe)
            db.session.flush()
            
            # save new recipe to database with the above details
            new_recipe = Recipe(
                name="please generate",
                cost=recipe_cost,
                time=recipe_time,
                serves=recipe_serves
            )
            
            # Add the new recipe to the database
            db.session.add(new_recipe)
            db.session.flush()  # To get the ID for the new recipe after adding it
            
            # create a string with the old cuisine and a request to not have the old recipe.name again
            new_recipe.cuisine = recipe_cuisine + f" : anything but {old_recipe.name}"
            
            # add ingredients to the new recipe
            add_available_pantry_items_to_recipe(new_recipe, user, meal_plan)

            # Associate the new recipe with the user
            user.recipes.append(new_recipe)

            # Associate the new recipe with the meal object
            meal.recipes.append(new_recipe)
            db.session.flush()

            # Associate the new recipe with the meal plan
            meal_plan.recipes.append(new_recipe)

            # generate the new recipe
            recipe_details = fetch_recipe_details(new_recipe.id, recipe_temperature)

            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
        return recipe_details

# celery task to generate a single recipe from scratch using the user's pantry
@celery.task
def generate_new_recipe(user_id, recipe_id):
    app = create_app_instance()
    with app.app_context():
        try:
            user = User.query.filter_by(id=user_id).first()
            recipe = Recipe.query.filter_by(id=recipe_id).first()
            profile = UserProfile.query.filter_by(user_id=user_id).first()
            recipe_temperature = profile.recipe_temperature or 1
            pantry = user.pantry

            # add text to the cuisine to request that only current pantry items are used
            if recipe.cuisine:
                recipe.cuisine = recipe.cuisine + " : Strictly only use items listed"
            else:
                recipe.cuisine = "Strictly only use items listed"
            if pantry:
                for pantry_item in pantry.pantry_items:
                    recipe_item = RecipeItem(recipe_id=recipe_id, item_id=pantry_item.item_id, quantity = pantry_item.quantity, unit = pantry_item.unit)
                    db.session.add(recipe_item)
            db.session.flush()

            # Associate the new recipe with the user
            user.recipes.append(recipe)

            
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
        # generate the new recipe
        recipe_details = fetch_recipe_details(recipe_id, recipe_temperature)
        return recipe_details

    
# Recipe generation; add available pantry items to a single recipe
def add_available_pantry_items_to_recipe(recipe, user, meal_plan):
    from mealmuse.utils import update_item_quantity
    # first add all items in pantry to the recipe
    pantry = user.pantry
    if pantry:
        for pantry_item in pantry.pantry_items:
            recipe_item = RecipeItem(recipe_id=recipe.id, item_id=pantry_item.item_id, quantity = pantry_item.quantity, unit = pantry_item.unit)
            db.session.add(recipe_item)
    db.session.flush()

    # for each item in the recipe that is also used in the curernt meal plan reduce the amount of the recipe item by the amount used in the meal plan using the update_item_quantity function
    # make a list of all the ingredients used in the meal plan
    meal_plan_ingredients = []
    for recipe in meal_plan.recipes:
        for recipe_item in recipe.recipe_items:
            meal_plan_ingredients.append(recipe_item)

    for recipe_item in recipe.recipe_items:
        for meal_plan_ingredient in meal_plan_ingredients:
            if recipe_item.item_id == meal_plan_ingredient.item_id:
                #subtract the mealplan ingredient quantity from the recipe item quantity
                quantity = -1 * meal_plan_ingredient.quantity              
                update_item_quantity(recipe_item, quantity, meal_plan_ingredient.unit)

    db.session.commit()


# Recipe generation: make a new recipe from using only a block of text passed in.
@celery.task
def create_recipe_with_text(recipe_text, user_id):
    app = create_app_instance()
    with app.app_context():
        try:
            user = User.query.filter_by(id=user_id).first()
            profile = UserProfile.query.filter_by(user_id=user_id).first()
            recipe_temperature = profile.recipe_temperature or 1

            # create a new recipe
            new_recipe = Recipe(
                name="please_process",
                description=recipe_text
            )
            db.session.add(new_recipe)
            db.session.flush()

            # Associate the new recipe with the user
            user.recipes.append(new_recipe)
            new_recipe_id = new_recipe.id
            db.session.commit()
            db.session.remove()
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise

        # generate the new recipe
        fetch_recipe_details(new_recipe_id, recipe_temperature)
        return new_recipe_id



# Meal Plan generation; create a full user prompt with flask app context
def create_meal_plan_user_prompt_with_context(user_id, meal_plan_id):
    app = create_app_instance()
    with app.app_context():
        try:
            user = User.query.filter_by(id=user_id).first()
            meal_plan = MealPlan.query.filter_by(id=meal_plan_id).first()
            user_prompt = create_meal_plan_user_prompt(user, meal_plan)
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
        return user_prompt
    

# Meal Plan generation; process the user input to create a user prompt in the expected format
def create_meal_plan_user_prompt(user, meal_plan):
    
    # Placeholder for the result in json format
    result = {}

    # get the user's pantry items
    pantry_items = []
    pantry = user.pantry
    if pantry:
        pantry_items = [item.item.name for item in pantry.pantry_items]

    # check if user has any equipment
    equipment = []
    if user.equipment:
        equipment = [equipment.name for equipment in user.equipment]

    # check if user has any allergies
    allergy = []
    if user.allergies:
        allergy = [allergy.name for allergy in user.allergies]

    # check if user has any dietary restrictions
    diet = []
    if user.diets:
        diet = [diet.name for diet in user.diets]

    # get the user's proficiency
    user_profile = UserProfile.query.filter_by(user_id=user.id).first()
    if user_profile:
        proficiency = user_profile.proficiency
    else:
        # create a profile and set proficiency to intermediate
        user_profile = UserProfile(user_id=user.id, proficiency="Beginner")
        db.session.add(user_profile)
        db.session.commit()
        proficiency = user_profile.proficiency

    # get the pantry use preference and budget jand leftover management for this meal plan
    pantry_usage_preference = meal_plan.pantry_use
    budget = meal_plan.budget
    leftovers = meal_plan.leftovers
    cuisine = meal_plan.cuisine_requests



    # Build the json object
    general = {
        "allergies": allergy,
        "cuisine and user requests": cuisine if cuisine else 'any', # Defaulting to 'Any' if not provided
        "dietary restrictions": diet if diet else 'no restrictions', # Defaulting to 'No restrictions' if not provided
        "pantry_items": pantry_items,
        "pantry_usage_preference": pantry_usage_preference,
        # "calorie_range": calorie_range,
        # "macronutrients": {
        #     "carbs": 45,    # You can replace with actual data if available
        #     "protein": 25,  # You can replace with actual data if available
        #     "fats": 30      # You can replace with actual data if available
        # },
        "equipment": equipment,
        "culinary_skill": proficiency,
        "budget": budget, 
        "meal_diversity": "high", #TO DO: meal_diversity,
        "leftover_management": leftovers,
        "description": "please generate"
    }

    daily = {}

    # meal_plan.days fetches the days associated with this meal plan
    for day in meal_plan.days:
        daily[day.name] = []
        for meal in day.meal:
            meal_details = {
                "name": meal.name,
                "prep_time": meal.prep_time,
                "num_people": meal.num_people,
                "cuisine": meal.cuisine,
                "type": meal.type
            }
            daily[day.name].append(meal_details)
    
    # Compile the result
    result = {
            "general": general,
            "daily": daily
        }

    return json.dumps(result)


# Meal Plan generation; the api call to get a meal plan
def fetch_meal_plan_from_api(meal_plan_id, user_id, temp=1):
    
    # Create the user prompt
    user_prompt = create_meal_plan_user_prompt_with_context(user_id, meal_plan_id)
    response = openai.ChatCompletion.create(
        model=MEAL_PLAN_MODEL,
        messages=[
            {"role": "system", "content": MEAL_PLAN_TASK},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3000,
        temperature=temp,
    )
    meal_plan_text = response.choices[0].message['content']

    try:
        # Attempt to parse the output as JSON
        meal_plan_json = json.loads(meal_plan_text)
    except json.JSONDecodeError:
        # If the output is not JSON, raise InvalidOutputFormat
        raise InvalidOutputFormat("Output is not valid JSON")
        
    return meal_plan_json
    # return meal_plan_output_gpt_4_v2

def save_meal_plan_output_with_context(meal_plan_json, meal_plan_id, user_id):
    app = create_app_instance()
    with app.app_context():
        try:

            meal_plan = MealPlan.query.filter_by(id=meal_plan_id).first()
            user = User.query.filter_by(id=user_id).first()
            save_meal_plan_output(meal_plan_json, meal_plan, user)
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
        return meal_plan_id
    

# Meal Plan Save: takes the output from the meal plan api call and saves it to the database
def save_meal_plan_output(meal_plan_json, meal_plan, user):

    # save the description to the meal plan
    meal_plan.description = meal_plan_json['description']

    for day_name, day_data in meal_plan_json['days'].items():
        # Find the day object corresponding to the day name (like 'Tuesday' or 'Wednesday')
        day_obj = Day.query.filter_by(name=day_name, meal_plan_id=meal_plan.id).first()
        if not day_obj:
            continue
        
        for meal_name, meal_data in day_data.items():
            # Find the meal object corresponding to the meal name (like 'Breakfast', 'Lunch'...)
            meal_obj = Meal.query.filter_by(name=meal_name, day_id=day_obj.id).first()
            if not meal_obj:
                continue
            
            # Extract recipe data
            recipe_data = meal_data['recipe']
            new_recipe = Recipe(
                name=recipe_data['name'],
                cost=recipe_data['cost_in_dollars'],
                time=recipe_data['time_required'],
                serves=recipe_data['serves'],
                cuisine=meal_obj.cuisine
            )
            
            # Add the new recipe to the database
            db.session.add(new_recipe)
            db.session.flush()  # To get the ID for the new recipe after adding it
            # Associate the new recipe with the meal object
            meal_obj.recipes.append(new_recipe)

            # Associate the new recipe with the meal plan
            meal_plan.recipes.append(new_recipe)

            # Associate the new recipe with the user
            user.recipes.append(new_recipe)

            # Save ingredients to the RecipeItem model
            for ingredient in recipe_data['ingredients_from_pantry']:
                # Here we're assuming each ingredient is a new unique item. If not, 
                # you'd need to check the database for existing items before creating a new one.
                item = Item(name=ingredient)
                db.session.add(item)
                db.session.flush()  # To get the ID for the new item after adding it

                # Create a RecipeItem instance
                recipe_item = RecipeItem(recipe_id=new_recipe.id, item_id=item.id)
                db.session.add(recipe_item)

            meal_plan.status = "complete"
            db.session.commit()

    return meal_plan.id


# Meal Plan generation; the api call to get a recipe
@celery.task
def fetch_recipe_details(recipe_id, temp=1):

    # for testing only
    # recipe = db.session.query(Recipe).filter_by(id=recipe_id).first()
    # recipe_name = recipe.name
    # db.session.close()
    retries = 2
    recipe_user_prompt = create_recipe_user_prompt(recipe_id)
    for _ in range(retries):
            ############################################ RECIPE API CALL ############################################
        response = openai.ChatCompletion.create(
            model=RECIPE_MODEL,
            messages=[
                {"role": "system", "content": RECIPES_TASK},
                {"role": "user", "content": recipe_user_prompt},
            ],
            max_tokens=2000,
            temperature=temp,
        )
        recipes_text = response.choices[0].message['content']

        # fake the api call for testing
        # recipes_text = get_recipe(recipe.name)

        try:
            return process_recipe_output(recipes_text, recipe_id)
        except InvalidOutputFormat as e:
            print(f"Error processing recipe for {recipe_id}: {e}. Retrying...")
            
    raise Exception(f"Failed to get a valid response for {recipe_id} after {retries} attempts.")

# Meal Plan generation; Pull info from db to create a user prompt for a recipe
def create_recipe_user_prompt(recipe_id):
    app = create_app_instance()
    with app.app_context():
        try:
            recipe = db.session.query(Recipe).filter_by(id=recipe_id).first()
            # Placeholder for the result in json format
            result = {}

            # check if this is a text-entry recipe that we are just processing
            if recipe.name == "please_process":
                recipe.name = "please generate"
                result = recipe.description
                recipe.description = ""
                db.session.commit()
            
            # otherwise assume generating a new recipe
            else:
                # get the recipe specific details
                name = recipe.name or "please generate"
                cost = recipe.cost or "any"
                time = recipe.time or "any"
                serves = recipe.serves or 1
                cuisine = recipe.cuisine or "be creative"

                # get the recipe's ingredients
                ingredients = []
                for recipe_item in recipe.recipe_items:
                    ingredients.append(recipe_item.item.name)

                # create text file with description and the above details
                result = {
                "recipe":{
                    "name": name,
                    "cost": cost,
                    "total time to make": time,
                    "serves": serves,
                    "ingredients from pantry to consider including": ingredients,
                    "cuisine or user requests": cuisine
                }}
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
    db.session.remove()
    return json.dumps(result)


# Meal Plan Save; takes the output from the recipe api call and saves it to the database
def process_recipe_output(data, recipe_id):
    app = create_app_instance()
    with app.app_context():
        try:
            recipe = db.session.query(Recipe).filter_by(id=recipe_id).first()
            
            # remove all existing ingredients from the recipe
            for recipe_item in recipe.recipe_items:
                db.session.delete(recipe_item)
            db.session.commit()
            
            # If data is a string, try to deserialize it as JSON
            if isinstance(data, str):
                try:
                    data = load_json_with_fractions(data)
                except json.JSONDecodeError:
                    print(f"invalid json: {data}")
                    raise InvalidOutputFormat("Provided string is not valid JSON")

            # Check if the data has 'recipe' key format
            if "recipe" not in data:
                print(f"no recipe: {data}")
                raise InvalidOutputFormat("Output does not have a 'recipe' key")
            
            details = data["recipe"]
            # Validating recipe name
            if recipe.name == "please generate":
                name = details.get('name')
                if not name or not isinstance(name, str):
                    print(f"no name: {name}")
                    raise InvalidOutputFormat("Missing or invalid name for recipe")
                recipe.name = name
            
            # Validating ingredients
            ingredients = details.get('ingredients', [])
            if not ingredients or not isinstance(ingredients, list):
                print(f"no ingredients: {ingredients}")
                raise InvalidOutputFormat("Missing or invalid ingredients for recipe")

            # Validate and save each ingredient
            for ingredient in ingredients:
                if not all(key in ingredient for key in ['name', 'quantity', 'unit']):
                    print(f"invalid ingredient: {ingredient}")
                    raise InvalidOutputFormat("Invalid ingredient format for recipe")
                
                # Check if the ingredient already exists in the database
                existing_item = db.session.query(Item).filter(Item.name == ingredient['name']).first()
                if existing_item:
                    item = existing_item
                else:
                    item = Item(name=ingredient['name'])
                    db.session.add(item)
                    db.session.flush()
                # Create a RecipeItem instance
                recipe_item = RecipeItem(recipe_id=recipe.id, item_id=item.id, quantity=ingredient['quantity'], unit=ingredient['unit'])
                db.session.add(recipe_item)
                db.session.flush()

            # Validating cooking instructions
            instructions = details.get('cooking_instructions', [])
            if not instructions or not isinstance(instructions, list):
                print(f"no instructions: {instructions}")
                raise InvalidOutputFormat("Missing or invalid cooking instructions for recipe")
 
            # add instructions to recipe
            recipe.instructions = "\n".join(instructions)

            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
    return recipe_id


# add a list of pantry items to the user's pantry
@celery.task
def add_list_of_items(user_id, list_of_items):
    pantry_items = process_list_of_items(list_of_items)
    save_pantry_list_to_db(pantry_items, user_id)
    return user_id


# api call to add a list of items to the user's pantry
def process_list_of_items(list_of_items):
    response = openai.ChatCompletion.create(
        model=MEAL_PLAN_MODEL,
        messages=[
            {"role": "system", "content": PANTRY_ITEMS_TASK},
            {"role": "user", "content": list_of_items},
        ],
        max_tokens=3000,
        temperature=1,
    )
    ingredient_list_text = response.choices[0].message['content']

    return ingredient_list_text


# process the list of items to be added to the user's pantry
def save_pantry_list_to_db(pantry_items, user_id):

    app = create_app_instance()
    with app.app_context():
        try:
             # If data is a string, try to deserialize it as JSON
            if isinstance(pantry_items, str):
                try:
                    pantry_items = load_json_with_fractions(pantry_items)
                except json.JSONDecodeError:
                    print(f"invalid json: {pantry_items}")
                    raise InvalidOutputFormat("Provided string is not valid JSON")
                
            # get the user and their pantry
            user = User.query.filter_by(id=user_id).first()
            pantry = user.pantry
            if not pantry:
                pantry = Pantry(user_id=user_id)
                db.session.add(pantry)
                db.session.flush()

            # Validating ingredients
            pantry_item_list = pantry_items.get('pantry_items', [])
            if not pantry_item_list or not isinstance(pantry_item_list, list):
                print(f"no ingredients: {pantry_item_list}")
                raise InvalidOutputFormat("Missing or invalid items for pantry")

            # Validate and save each ingredient
            for pantry_item in pantry_item_list:
                if not all(key in pantry_item for key in ['name', 'quantity', 'unit']):
                    print(f"invalid ingredient: {pantry_item}")
                    raise InvalidOutputFormat("Invalid pantry item format")
                
                # Check if the ingredient already exists in the database
                existing_item = db.session.query(Item).filter(Item.name == pantry_item['name']).first()
                if existing_item:
                    item = existing_item
                else:
                    item = Item(name=pantry_item['name'])
                    db.session.add(item)
                    db.session.flush()
                # Create a PantryItem instance if one does not already exist otherwise increase the quantity
                existing_pantry_item = db.session.query(PantryItem).filter(PantryItem.pantry_id == pantry.id, PantryItem.item_id == item.id).first()
                if existing_pantry_item:
                    existing_pantry_item.quantity += pantry_item['quantity']
                    existing_pantry_item.unit = pantry_item['unit']
                else:
                    new_pantry_item = PantryItem(pantry_id=pantry.id, item_id=item.id, quantity=pantry_item['quantity'], unit=pantry_item['unit'])
                    db.session.add(new_pantry_item)
                    db.session.flush()

            db.session.commit()

        except Exception as e:
            db.session.rollback()
            print(f"Error occurred: {e}")
            db.session.remove()
            raise
    return user_id


def fraction_to_decimal(match):
    """Converts a fraction to its decimal representation."""
    num, den = map(int, match.group(0).split('/'))
    return str(num / den)

def preprocess_json_string(s):
    """Replaces fractions with their decimal equivalents in a string."""
    return re.sub(r'\b\d+/\d+\b', fraction_to_decimal, s)

def load_json_with_fractions(s):
    """Loads a JSON string, even if it contains fractions."""
    preprocessed_string = preprocess_json_string(s)
    return json.loads(preprocessed_string)


# from celery.signals import worker_process_init

# @worker_process_init.connect
# def on_worker_init(*args, **kwargs):
#     warmup.apply_async()



# @celery.task
# def warmup():
#     # Perform some simple database queries
#     some_query = db.session.query(Recipe).limit(1)
#     db.session.remove()
#     some_query = db.session.query(Recipe).limit(1)
#     another_query = db.session.query(User).limit(1)
#     query_three = db.session.query(MealPlan).limit(1)
#     query_four = db.session.query(Day).limit(1)
#     query_five = db.session.query(Meal).limit(1)
#     query_six = db.session.query(Pantry).limit(1)
#     query_seven = db.session.query(Item).limit(1)
#     query_eight = db.session.query(ShoppingList).limit(1)
#     query_nine = db.session.query(RecipeItem).limit(1)
#     query_ten = db.session.query(ShoppingListItem).limit(1)
#     query_eleven = db.session.query(PantryItem).limit(1)
#     query_twelve = db.session.query(UserProfile).limit(1)
#     query_thirteen = db.session.query(Equipment).limit(1)
#     query_fourteen = db.session.query(Allergy).limit(1)
#     query_fifteen = db.session.query(Diet).limit(1)
#     query_sixteen = db.session.query(users_recipes).limit(1)
#     query_seventeen = db.session.query(recipes_mealplans).limit(1)
#     query_eighteen = db.session.query(recipes_meals).limit(1)

#     # Close the session
#     db.session.remove()

#     # run celery worker with the following command: "celery -A mealmuse.tasks worker --loglevel=info"
