import openai
import psycopg2

import postgresconfig
import openaiconfig

# Set your API key
api_key = openaiconfig.openaikey

# Initialize the OpenAI API client
openai.api_key = api_key

connection = psycopg2.connect(
    database=postgresconfig.database,
    user=postgresconfig.user,
    password=postgresconfig.password,
    host=postgresconfig.host,
    port=postgresconfig.port,
)
cursor = connection.cursor()
print("Connected to the database!")

# get all recipes
cursor.execute(
    "SELECT ID, recipe_name, description FROM recipes WHERE categorized = false ORDER BY id ASC LIMIT 100;"
)
recipes = cursor.fetchall()

# create a list of dictionaries
recipe_list = []
for recipe in recipes:
    recipe_dict = {"id": recipe[0], "name": recipe[1], "description": recipe[2]}
    recipe_list.append(recipe_dict)

# get all categories
cursor.execute("SELECT id, category FROM recipe_categories;")

categories = cursor.fetchall()

# create a list of dictionaries
category_list = []
for category in categories:
    category_dict = {"id": str(category[0]), "name": category[1]}
    category_list.append(category_dict)

formatted_categories = [
    f"ID: {category['id']} - Name: {category['name']}" for category in category_list
]

category_string = ", \n".join(formatted_categories)

print("Recipes and categories fetched from the database!")


# Function to categorize a recipe
def categorize_recipe(recipe):
    try:
        # Call the OpenAI API to categorize the recipe based on the prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f'You are a helpful assistant that is great at categorizing recipes. You know these categories, they are in the format <ID: "NAME">.\n\nWhen you are provided with a recipe name and description, you will choose one or more categories from the list. If the recipe name includes vegan you can assume the Vegan category will be chosen. Output them into a postgres query to insert into a table called recipes_to_categories linking table, for example:\n\nINSERT INTO recipes_to_categories (recipe_id, category_id)\nVALUES\n    (61, 10), -- Spinach falafel & hummus bowl (Category: Vegan)\n    (61, 22), -- Spinach falafel & hummus bowl (Category: Quick and Easy)\n    (61, 27); -- Spinach falafel & hummus bowl (Category: Slow Cooker)\nDo not leave trialing commas\n\nCategory List:\n {category_string}",',
                },
                {
                    "role": "user",
                    "content": f"ID: {recipe['id']}, Recipe Name: {recipe['name']}, Description: {recipe['description']}",
                },
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        insert_query = response.choices[0].message.content.strip()

        print(insert_query)

        # insert into the database
        cursor.execute(insert_query)

        # update the categorized column to true
        cursor.execute(
            f"UPDATE recipes SET categorized = true WHERE id = {str(recipe['id'])};"
        )

        # commit the changes
        connection.commit()
    except Exception as e:
        print(e)
        connection.rollback()
        categorize_recipe(recipe)


# categorize_recipe(recipe_list[0])

# Loop through your recipes and categorize them
for recipe in recipe_list:
    print('Categorizing recipe: "' + recipe["name"] + '"')
    categorize_recipe(recipe)
