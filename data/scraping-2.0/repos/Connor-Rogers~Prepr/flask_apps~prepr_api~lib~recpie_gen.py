import openai
import random
from lib.firebase import db
from decouple import config


def get_random_recipe_from_likes(user_id):
    """
    Retrieves a random recipe that the given user has liked.
    If the user hasn't liked any recipes, returns any random recipe from the database.

    :param user_id: ID of the user.
    :return: A dictionary containing the recipe information or an empty dictionary if no recipe found.
    """

    # Find recipes liked by the user
    liked_recipes = []
    likes_docs = db.collection("likes").stream()

    for doc in likes_docs:
        likes = doc.to_dict().get("likes", [])
        if user_id in likes:
            liked_recipes.append(doc.id)

    # If the user has liked any recipes, pick one at random
    if liked_recipes:
        random_recipe_id = random.choice(liked_recipes)
    else:
        # If user hasn't liked any recipe, pick a random recipe from all recipes
        all_recipe_ids = [doc.id for doc in db.collection("recipes").stream()]
        if not all_recipe_ids:
            return {}  # Return empty dict if there are no recipes in the database
        random_recipe_id = random.choice(all_recipe_ids)

    # Get the recipe details from the "recipes" collection
    recipe_doc = db.collection("recipes").document(random_recipe_id).get()
    if recipe_doc.exists:
        return recipe_doc.to_dict()
    return {}


def generate_recipe(user_id, pantry_items: list = None):
    """
    Generates a new recipe using OpenAI based on a random recipe the user has liked and ingredients from the pantry.

    :param user_id: ID of the user.
    :param pantry_items: List of items present in the user's pantry. Default is None.
    :return: A string containing the generated recipe in the specified format.
    """

    openai.api_key = config("OPENAI_KEY")
    items_for_day = []
    if pantry_items:
        num_to_sample = random.randint(0, len(pantry_items))
        if num_to_sample == 0:
            num_to_sample = 1
        num_items = min(random.randint(1, num_to_sample), len(pantry_items))
        items_for_day = random.sample(pantry_items, num_items)
        pantry_items = [item for item in pantry_items if item not in items_for_day]
        prompt = (
            """Please generate a detailed recipe including the title, ingredients, instructions, and macro nutritional information. Use the following recipe as a template and inspiration """
            + str(get_random_recipe_from_likes(user_id))
            + """ incorporate the following additional ingredients into the recipe """
            + str(items_for_day)
            + """ and should be formated as a json with the following feilds: 
    title: "title of recipe"
    ingredients: [{"ingredient": "ingredient name", "quantity": "quantity unit"}, {"ingredient": "ingredient name", "quantity": "quantity unit"}]
    instructions: "Instructions for recipe"
    calories: "calories"`
    fats: "fats"
    carbs: "carbs"
    proteins: "proteins"
    """
        )
    else:
        prompt = (
            """Please generate a detailed recipe including the title, ingredients, instructions, and macro nutritional information. Use the following recipe as a template and inspiration """
            + str(get_random_recipe_from_likes())
            + """ and should be formated as a json with the following feilds: 
    title: "title of recipe"
    ingredients: [{"ingredient": "ingredient name", "quantity": "quantity unit"}, {"ingredient": "ingredient name", "quantity": "quantity unit"}]
    instructions: "Instructions for recipe"
    calories: "calories"`
    fats: "fats"
    carbs: "carbs"
    proteins: "proteins"
    """
        )

    # Requesting the OpenAI API to generate the recipe
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=600,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    recipe = response.choices[0].text.strip()
    return recipe
