"""
This file defines actions, i.e. functions the URLs are mapped into
The @action(path) decorator exposes the function at URL:

    http://127.0.0.1:8000/{app_name}/{path}

If app_name == '_default' then simply

    http://127.0.0.1:8000/{path}

If path == 'index' it can be omitted:

    http://127.0.0.1:8000/

The path follows the bottlepy syntax.

@action.uses('generic.html')  indicates that the action uses the generic.html template
@action.uses(session)         indicates that the action uses the session
@action.uses(db)              indicates that the action uses the db
@action.uses(T)               indicates that the action uses the i18n & pluralization
@action.uses(auth.user)       indicates that the action requires a logged-in user
@action.uses(auth)            indicates that the action requires the auth object

session, db, T, auth, and templates are examples of Fixtures.
Warning: Fixtures MUST be declared with @action.uses({fixtures}) else your app will result in undefined behavior
"""

import re
import json
from py4web.utils.form import Form, FormStyleBulma
from py4web import action, request, abort, redirect, URL, HTTP
from yatl.helpers import A
from .common import (
    db,
    session,
    auth,
)

from datetime import datetime
from py4web.utils.url_signer import URLSigner
from .models import get_user_email

import openai
from dotenv import dotenv_values

secrets = dotenv_values("apps/RecipeWizard/.env")

url_signer = URLSigner(session)


@action("index")
@action.uses("index.html", db, auth.user, url_signer)
def index():
    response = dict(
        # COMPLETE: return here any signed URLs you need.
        getPantry_url=URL("getPantry", signer=url_signer),
        addItemToPantry_url=URL("addItemToPantry", signer=url_signer),
        deleteItem_url=URL("deleteItem", signer=url_signer),
        generateRecipeSuggestion_url=URL("generateRecipeSuggestion"),
        getRecipes_url=URL("getRecipes", signer=url_signer),
        deleteRecipe_url=URL("deleteRecipe", signer=url_signer),
        favRecipe_url=URL("favRecipe", signer=url_signer),
        getFavs_url=URL("getFavs", signer=url_signer),
        deleteFav_url=URL("deleteFav", signer=url_signer),
        togglePin_url=URL("togglePin", signer=url_signer),
        getPinned_url=URL("getPinned", signer=url_signer),
        uploadImage_url=URL("upload_image", signer=url_signer),
        getUserID_url=URL("getUserID", signer=url_signer),
        setPinnedRecipeImageURL_url=URL(
            "setPinnedRecipeImageURL", signer=url_signer),
    )
    return response


@action("getPantry", method="GET")
@action.uses(db, auth.user, url_signer)
def getPantry():
    response = dict(pantry=db(db.pantry.user_id ==
                    auth.current_user.get("id")).select().as_list())
    return response


@action("addItemToPantry", method="POST")
@action.uses(db, auth.user, url_signer)
def addItemToPantry():
    userID = auth.current_user.get("id")
    item = request.json.get("item")
    if db((db.pantry.user_id == userID) & (db.pantry.item == item)).select().first():
        response = dict(success=False)
    else:
        db.pantry.insert(user_id=userID, item=item)
        newItem = db((db.pantry.user_id == userID) & (
            db.pantry.item == item)).select().first()
        response = dict(success=True, newItem=newItem)
    return response


@action("deleteItem", method="POST")
@action.uses(db, auth.user, url_signer)
def deleteItem():
    itemID = request.json.get("itemID")
    db(db.pantry.id == itemID).delete()
    return dict()


defaultPrompt = """
{
  "instructions": "Given a list of ingredients and user preferences, generate a recipe suggestion that meets all the following criteria:",
  "criteria": [
    "Utilize the provided ingredients exclusively to reduce food waste and maximize resourcefulness.",
    "Offer a variety of recipe options, including breakfast, lunch, dinner, snacks, and desserts, to cater to different meal preferences.",
    "The generated recipe suggestion does not need to include all pantry items. Use a subset of the pantry items to create a reasonable yummy recipe.",
    "Provide a recipe that is not included in the given list of existing recipes.",
    "Optionally, consider recipes that are quick and easy to prepare, perfect for busy individuals or those with limited cooking time.",
    "Optionally, provide recipes with a balanced nutritional profile, considering macronutrients and minimizing sugar content."
  ],
  "instructionsNote": "Please tap into your culinary expertise and creativity to generate diverse, delicious, and practical recipe suggestions. Assume the provided ingredients are available in sufficient quantities. If necessary, you can make reasonable assumptions about ingredient preparation techniques (e.g., chopping, cooking methods).",
  "examples": [
    {
      "ingredients": "[List the ingredients]",
      "numberOfPeople": "[Specify the number of people the user is cooking for]"
    }
  ],
  "prompt": "Please generate a single recipe based on the provided information.",
  "userInput": "[Provide the list of ingredients and specify the dietary preferences and restrictions, as well as the number of people cooking for.]",
  "rule": "Meat-based options should be included when \\"NONE\\" is specified as the dietary preference. The AI will generate recipe suggestions that include both meat-based and vegetarian options."
}
"""
# Convert the prompt to a JSON string
prompt_json = json.loads(defaultPrompt)

# Use prompt_json in your code as needed
# For example:
# print(prompt_json["instructions"])  # Output: Given a list of ingredients and user preferences, generate a recipe suggestion that meets all the following criteria:


def split_recipe_string(recipe):
    # Regular expressions to match the parts
    title_re = r'Recipe Suggestion:\n*(.+?)\nIngredients:'
    ingredients_re = r'Ingredients:\n(.+?)Instructions:'
    instructions_re = r'Instructions:\n(.+)'

    # Match the parts
    title_match = re.search(title_re, recipe, re.DOTALL)
    ingredients_match = re.search(ingredients_re, recipe, re.DOTALL)
    instructions_match = re.search(instructions_re, recipe, re.DOTALL)

    # Parse the parts
    title = title_match.group(1).strip() if title_match else None
    ingredients = ingredients_match.group(1).strip().split(
        '\n') if ingredients_match else None
    instructions = instructions_match.group(1).strip().split(
        '\n') if instructions_match else None

    return {
        'title': title,
        'ingredients': ingredients,
        'instructions': instructions
    }


def getExistingRecipeTitles():
    userID = auth.current_user.get("id")
    recipes = db(db.recipes.created_by == userID).select().as_list()
    titles = []
    for recipe in recipes:
        titles.append(recipe["title"])
    return titles


@action("generateRecipeSuggestion", method="GET")
@action.uses(db, auth.user)
def generateRecipeSuggestion():
    # print("\nCalling a recipe suggestion generation!")
    openai.api_key = secrets["OPENAI_KEY"]

    userID = auth.current_user.get("id")
    ingredients = db(db.pantry.user_id == userID).select().as_list()
    numberOfPeople = 3  # TODO in future want to pull from URL
    existingRecipes = getExistingRecipeTitles()

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f'{json.dumps(prompt_json)} Ingredients: {json.dumps(ingredients)}, Existing Recipes: {json.dumps(existingRecipes)}, Number of People: {numberOfPeople}',
        max_tokens=300,
        temperature=0.3,
    )

    # Store the recipe text in a variable
    recipe_text = response.choices[0].text.strip()

    split_recipe = split_recipe_string(recipe_text)
    # Print out the separate parts
    # print("Title:", split_recipe["title"])
    # print("Ingredients:", split_recipe["ingredients"])
    # print("Instructions:", split_recipe["instructions"])

    # Save the recipe in the database
    recipe_id = db.recipes.insert(
        created_by=userID,
        title=split_recipe["title"],
        ingredients=split_recipe["ingredients"],
        instructions=split_recipe["instructions"],
    )

    # print(recipe_id)
    # Return the recipe JSON as the response
    return dict(recipe={
        "id": recipe_id,
        "title": split_recipe["title"],
        "ingredients": split_recipe["ingredients"],
        "instructions": split_recipe["instructions"],
    })


@action("getRecipes", method="GET")
@action.uses(db, auth.user, url_signer)
def getRecipes():
    userID = auth.current_user.get("id")
    recipes = db(db.recipes.created_by == userID).select(
        db.recipes.id, db.recipes.title, db.recipes.ingredients,
        db.recipes.instructions).as_list()
    # print(recipes)

    return dict(recipes=recipes)


@action("deleteRecipe", method="POST")
@action.uses(db, auth.user, url_signer)
def deleteRecipe():
    recipeID = request.json.get("recipeID")
    # print(f"Deleting recipe with ID {recipeID}")
    status = db(db.recipes.id == recipeID).delete()
    # print("status:", status)
    return dict(status=status)


@action("deleteFav", method="POST")
@action.uses(db, auth.user, url_signer)
def deleteFav():
    favID = request.json.get("favID")
    # print(f"Deleting recipe with ID {recipeID}")
    status = db(db.favorites.id == favID).delete()
    # print("status:", status)
    return dict(status=status)


@action("favRecipe", method="POST")
@action.uses(db, auth.user, url_signer)
def favRecipe():
    userID = auth.current_user.get("id")
    recipeTitle = request.json.get("recipeTitle")
    if recipeTitle is None or "":
        recipeTitle = "Unnamed"
    recipeIngredients = request.json.get("recipeIngredients")
    if recipeIngredients is None or "":
        recipeIngredients = "No ingredients provided"
    recipeInstructions = request.json.get("recipeInstructions")
    if recipeInstructions is None or "":
        recipeInstructions = "No instructions provided"
    # print("Request to favorite recipe: ", recipeTitle)
    # Check if recipeTitle is already in favoritesDB
    existingFav = db(db.favorites.user_id == userID).select().as_list()
    if existingFav is not None:
        for fav in existingFav:
            if fav["title"] == recipeTitle:
                # print("Recipe already favorited")
                return dict(success=False)

    db.favorites.insert(
        user_id=userID,
        title=recipeTitle,
        ingredients=recipeIngredients,
        instructions=recipeInstructions,
    )
    # print("Success")
    return dict(success=True)


@action("getFavs", method="GET")
@action.uses(db, auth.user, url_signer)
def getFavs():
    userID = auth.current_user.get("id")
    favorites = db(db.favorites.user_id == userID).select().as_list()
    # print("Returning Favorites", favorites)
    return dict(favorites=favorites)


@action("togglePin", method="POST")
@action.uses(db, auth.user, url_signer)
def togglePin():
    userID = auth.current_user.get("id")
    favID = request.json.get("favID")

    # Grab the favorite recipe row
    favRecipe = db((db.favorites.id == favID) & (
        db.favorites.user_id == userID)).select().first()

    # If the recipe is already pinned, unpin it
    if favRecipe.pinned:
        db(db.favorites.id == favID).update(pinned=False)
    # Otherwise unpin all other recipe
    else:
        db(db.favorites.user_id == userID).update(pinned=False)
        db(db.favorites.id == favID).update(pinned=True)
    return dict(success=True, pinnedRecipe=favRecipe)


@action("getPinned", method="GET")
@action.uses(db, auth.user, url_signer)
def getPinned():
    pinned_recipes = db(db.favorites.pinned == True).select()
    # Replace the userID's with the user's first name
    pinned_list = []
    for recipe in pinned_recipes:
        user = db(db.auth_user.id == recipe.user_id).select().first()
        first_name = user.first_name
        recipe_dict = {
            "dbID": recipe.id,
            "title": recipe.title,
            "ingredients": recipe.ingredients,
            "instructions": recipe.instructions,
            "favorited_at": recipe.favorited_at,
            "pinned": recipe.pinned,
            "user_name": first_name,
            "user_id": recipe.user_id,
            "imageUrl": recipe.imageUrl,
        }
        pinned_list.append(recipe_dict)
    # print("Returning Pinned!", pinned_list)
    return dict(pinned=pinned_list)


@action("getUserID", method="GET")
@action.uses(db, auth.user, url_signer)
def getUserID():
    userID = auth.current_user.get("id")
    return dict(userID=userID)


@action("setPinnedRecipeImageURL", method="POST")
@action.uses(db, auth.user, url_signer)
def setPinnedRecipeImageURL():
    # print("Setting pinned recipe image URL")
    dbID = request.json.get("dbID")
    imageUrl = request.json.get("imageUrl")
    # print("Requested Image URL:", imageUrl)
    # print("Updating recipe with ID", dbID)
    db(db.favorites.id == dbID).update(imageUrl=imageUrl)
    return dict(success=True)
