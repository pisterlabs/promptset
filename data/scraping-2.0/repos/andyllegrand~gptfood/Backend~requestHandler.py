import json
import openai
import sqlite3
import random
import requests
import logging

from customError import CustomError

RECIPES_PER_REQUEST = 5
IMAGE_DIR = '/Users/andylegrand/xcode/gptfood/Backend/tests/images/'

debug = True  # if set to true the backend will not call the openai api and will instead return example responses

errorCodes = json.loads(open('errorCodes.json', 'r').read())

def genRecipesApiCall(ingredients, usedRecipes, proomptPath='proomps/genRecipeList.txt'):
  """
  Calls openai api to generate recipes given ingredients.
  @param ingredients: list of ingredients representing available ingredients
  @param usedRecipes: list of recipes representing recipes that have already been used
  @raise CustomError: if the response from the api is not valid json
  @return: extracted text from response
  """

  # Form list of ingredients in string form
  ingredient_string = ''
  for ingredient in ingredients:
      ingredient_string += ingredient + '\n'

  # Form list of used recipes in string form
  used_recipe_string = ''
  for recipe in usedRecipes:
    used_recipe_string += '{' + recipe + '}\n'

  # Form proompt
  proompt = open(proomptPath, 'r').read()
  proompt = proompt.replace('[ingredients]', ingredient_string)
  proompt = proompt.replace('[used]', used_recipe_string)

  # Call openai api
  openai.api_key = open('/Users/andylegrand/xcode/gptfood/Backend/key.txt', 'r').read()
  logging.debug("key: " + openai.api_key)

  try:
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=proompt,
      temperature=1,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    assert response.choices[0].text != None
    
    logging.debug(f"APIResponse: {response.choices[0].text}")

    return response.choices[0].text
  except:
    raise CustomError(proompt, errorCodes["GPT_API_ERROR"])
  
def addRecipeToDatabase(recipe, ingredients, connection):
  """
  Adds a list of recipes to the database.
  @param recipes: string representing the recipe
  @param ingredients: list of strings representing the ingredients
  @param connection: connection to the database
  @return: None
  """
  cursor = connection.cursor()

  # Add the recipe to the database
  cursor.execute("""
    INSERT INTO recipes (name, directions, imagePath) VALUES (?, NULL, NULL);
  """, (recipe,))
  recipeId = cursor.lastrowid

  # Add the ingredients to the database
  ingredientIds = []
  for ingredient in ingredients:
    cursor.execute("""
      INSERT OR IGNORE INTO ingredients (name) VALUES (?);
    """, (ingredient,))
    ingredientIds.append(cursor.lastrowid)

  # Add the relations to the database
  for ingredientId in ingredientIds:
      cursor.execute("""
        INSERT INTO relations (recipe_id, ingredient_id) VALUES (?, ?);
      """, (recipeId, ingredientId))
  
  connection.commit()

def generateAndAddRecipes(ingredients, usedRecipes, connection):
  """
  Generates recipes and adds them to the database.
  @param ingredients: list of ingredients representing available ingredients
  @param usedRecipes: list of recipes representing recipes that have already been used
  @param connection: connection to the database
  @raise CustomError: if the response from the api is not valid json
  @return: None
  """
  completionText = None
  if not debug:
    completionText = genRecipesApiCall(ingredients, usedRecipes)
  else:
    completionText = open('sampleresponse.txt', 'r').read()
  
  # Load the text as JSON, abort and throw an error if it fails
  try:
    recipes = json.loads(completionText)
  except:
    raise CustomError(f"Error parsing JSON: {completionText}", errorCodes["JSON_PARSE_ERROR"])


  for recipe in recipes:
    addRecipeToDatabase(recipe["name"], recipe["ingredients"], connection)

def queryDatabaseRecipes(ingredients, usedRecipes, connection):
    cursor = connection.cursor()

    # Fetch all recipe names from the database, then randomize the order
    cursor.execute('SELECT name FROM recipes')
    all_recipes = [row[0] for row in cursor.fetchall()]
    random.shuffle(all_recipes)

    # Fetch corresponding ingredient ids from the database
    placeholders = ', '.join('?' for ingredient in ingredients)
    cursor.execute(f"SELECT id FROM ingredients WHERE name IN ({placeholders})", ingredients)
    ingredient_ids = set(row[0] for row in cursor.fetchall())

    # Find the recipes whose ingredients are all in the provided list
    matching_recipes = []
    for recipe in all_recipes:
        if recipe in usedRecipes:
            continue

        # Fetch the ingredients for this recipe from relations
        cursor.execute('''
            SELECT ingredient_id
            FROM relations
            JOIN recipes ON relations.recipe_id = recipes.id
            WHERE recipes.name = ?
        ''', (recipe,))
        recipe_ingredient_ids = set(row[0] for row in cursor.fetchall())

        # Loop through the ingredients and check if they are all in the provided list. If so, add the recipe to matching_recipes
        if recipe_ingredient_ids.issubset(ingredient_ids):
            matching_recipes.append(recipe)

        if len(matching_recipes) == RECIPES_PER_REQUEST:
            break

    return matching_recipes

def getRecipes(ingredients, usedRecipes, databasePath):
  if debug:
    return ["Recipe 1", "Recipe 2", "Recipe 3", "Recipe 4", "Recipe 5"]

  # Connect to the database
  conn = sqlite3.connect(databasePath)

  # Query database. If there are not enough recipes to fufill the request generate more and try again
  recipes = queryDatabaseRecipes(ingredients, usedRecipes, conn)
  
  if len(recipes) < RECIPES_PER_REQUEST:
    generateAndAddRecipes(ingredients, usedRecipes, conn)
    recipes = queryDatabaseRecipes(ingredients, usedRecipes, conn)
  
  conn.close()
  return recipes

def genDirectionsApiCall(recipe, ingredients):
  """
  Calls openai api to generate directions given a recipe.
  @param recipe: string representing the recipe
  @return: text response from openai
  """

  # Form list of ingredients in string form
  ingredient_string = ''
  for ingredient in ingredients:
      ingredient_string += ingredient + '\n'

  # Form proompt
  proompt = open('proomps/genDirections.txt', 'r').read()
  proompt = proompt.replace('[recipe]', recipe)
  proompt = proompt.replace('[ingredients]', ingredient_string)

  # Call openai api
  openai.api_key = open('key.txt', 'r').read()

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=proompt,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response.choices[0].text

def downloadImage(url, path):
  """
  Downloads an image from a url and saves it to a path.
  @param url: url of the image
  @param path: path to save the image to
  @return: None
  """
  response = requests.get(url, stream=True)
  if response.status_code == 200:
      with open(path, 'wb') as f:
          f.write(response.content)

def genImageApiCall(description):
  """
  Calls openai api to generate an image given a description.
  @param description: string representing the description
  @return: url of the generated image
  """
  openai.api_key = open('key.txt', 'r').read()

  response = openai.Image.create(
    prompt=description,
    n=1,
    size="256x256"
  )
  image_url = response['data'][0]['url']
  return image_url

def addDirectionsToDatabase(recipe, directions, imagePath, connection):
  """
  Adds directions and image path to the database.
  @param recipe: string representing the recipe
  @param directions: string representing the directions
  @param imagePath: string representing the path to the image
  @param connection: connection to the database
  @return: None
  """

  cursor = connection.cursor()

  # Add directions and image path to database
  cursor.execute("""
    UPDATE recipes SET directions = ?, imagePath = ? WHERE name = ?;
  """, (directions, imagePath, recipe))

  connection.commit()


def generateAndAddDirections(recipe, connection, imagePath):
  """
  Generates directions and adds them to the database.
  @param recipe: string representing the recipe
  @param connection: connection to the database
  @return: None
  """

  # Get ingredients from database
  cursor = connection.cursor()

  # get recipe id
  cursor.execute('SELECT id FROM recipes WHERE name = ?', (recipe,))

  if cursor.fetchone() == None:
    raise CustomError(f"Recipe {recipe} not found in database", errorCodes["RECIPE_NOT_FOUND"])

  # get ingredients
  cursor.execute('''
    SELECT ingredients.name
    FROM ingredients
    JOIN relations ON ingredients.id = relations.ingredient_id
    JOIN recipes ON relations.recipe_id = recipes.id
    WHERE recipes.name = ?
  ''', (recipe,))
  ingredients = [row[0] for row in cursor.fetchall()]

  logging.debug(f"Ingredients: {ingredients}")

  # Add the directions to the database
  res = genDirectionsApiCall(recipe, ingredients)

  # Convert to json, extract directions and image proompt. Abort and throw an error if it fails
  try:
    js = json.loads(res)
    directions = js["directions"]
    imageProompt = js["dall-e prompt"]
  except:
    raise CustomError(f"Error parsing JSON: {res}", errorCodes["JSON_PARSE_ERROR"])

  # Generate image
  imageUrl = genImageApiCall(imageProompt)
  imagePath = imagePath + recipe + '.png'
  downloadImage(imageUrl, imagePath)

  # Add directions and image path to database
  addDirectionsToDatabase(recipe, directions, imagePath, connection)

def getDirections(recipe, databasePath):
  """
  Query database. If the current recipe does not have directions generate them and try again
  @param recipe: string representing the recipe
  @param databasePath: path to the database
  @return: directions for the recipe
  """
  if debug:
    return "Directions for " + recipe

  # Connect to the database
  conn = sqlite3.connect(databasePath)
  cursor = conn.cursor()

  # Check if recipe has directions
  cursor.execute('SELECT directions FROM recipes WHERE name = ?', (recipe,))
  directions = cursor.fetchone()[0]

  if directions == None:
    generateAndAddDirections(recipe, conn, IMAGE_DIR)
    cursor.execute('SELECT directions FROM recipes WHERE name = ?', (recipe,))
    directions = cursor.fetchone()[0]

  conn.close()
  return directions


def getImage(recipe, databasePath):
  """
  Returns path to the image for the recipe. This function should be called after getDirections, so the image should already be generated.
  @param recipe: string representing the recipe
  @param databasePath: path to the database
  @return: path to the image
  """
  if debug:
     return "/Users/andylegrand/xcode/gptfood/Backend/exampleResponses/exampleImage.png"

  # Connect to the database
  conn = sqlite3.connect(databasePath)
  cursor = conn.cursor()

  # Check if recipe has directions
  cursor.execute('SELECT imagePath FROM recipes WHERE name = ?', (recipe,))
  imagePath = cursor.fetchone()[0]
  if imagePath == None:
    raise CustomError(f"Image for {recipe} not found in database", errorCodes["IMAGE_NOT_FOUND"])

  conn.close()
  return imagePath
