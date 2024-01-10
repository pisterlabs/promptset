import re
import json
import os
import openai
import webbrowser


def get_minutes(mseconds):
    return mseconds / 60000000.0


def read_recipe(file):
    y = json.loads(file.read())

    for x in range(4):
        print('')
    print(y['name'])
    print(f'Serves: {y["servings"]}')
    print('------')
    if 'cookTime' in y:
        print(f'Cooking time: {get_minutes(y["cookTime"])}')
    if 'prepTime' in y:
        print(f'Prep time: {get_minutes(y["prepTime"])}')
    print('------')
    print('Ingredients')
    for ingredient in y['ingredients']:
        print(f'-- {ingredient["name"]}')
    print('------')
    print('Steps')
    index = 0
    for step in y['instructions']:
        index += 1
        print(f'{index}) {step["text"]}')
    for x in range(4):
        print('')


def parse_ingredient(ingredient):
    match = re.search(
        r'\b(?:\d+\.?\d*|\d*\.?\d+)\s*[a-zA-Z]*\s*([a-zA-Z\- ]+)',
        ingredient)
    if match:
        return match.group(1).strip()
    return None


def get_config(key):
    with open('config', 'r') as config_file:
        config = config_file.read()
        for line in config.split('\n'):
            if line.startswith(key):
                return line.split(' : ')[1]


def gather_ingredients(files):
    ingredients = []

    for recipe in files:
        with open(recipe, 'r') as file:
            y = json.loads(file.read())

            for ingredient in y['ingredients']:
                ingredients.append(ingredient['name'])

    key = get_config('openai_api')

    if len(key) > 0:
        openai.api_key = get_config('openai_api')
    else:
        print('API key not found')
        return ingredients

    outingredients = ''

    for ingredient in ingredients:
        outingredients = outingredients + ingredient + '\n'

    city = ''
    city = get_config('city')

    prompt = ''

    if len(city) == 0:
        prompt = f'''
            Ingredients:
            {outingredients}

            Tasks:
            1. Merge like-items and convert measurements.
            2. Format as:
                **CATEGORY**
                [INGREDIENT]: [QUANTITY]
        '''
    else:
        prompt = f'''
            Ingredients:
            {outingredients}

            Tasks:
            1. Merge like-items and convert measurements.
            2. Recommend substitutes for {city} availability
                a. Substitutions should take up one line per substitution suggestion
            3. Format as:
                **CATEGORY**
                [INGREDIENT]: [QUANTITY]
        '''

    messages = [
        {"role": "user", "content": "You are a professional grocery shopper, making the most efficient, time-saving lists in the whole world. Remain brief and highly-efficient."},
        {"role": "user", "content": "Only use these categories: Produce, Canned Goods, Dairy, Meat, Deli, Seafood, Condiments & Spices, Bakery, Grains"},
        {"role": "user", "content": prompt}
    ]

    max_tokens = int(get_config('tokens'))

    response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
            temperature=0.5,
            max_tokens=max_tokens
        )

    print(response.choices[0].message.content)

    return response.choices[0].message.content.split('\n')


# Open the recipe card through 'JustTheRecipe.com'
def open_card(files):
    # For each of the files that the user's requested, open in read-only mode, load the JSON, find the 'sourceURL' value, and open in the browser
    for recipe in files:
        with open(recipe, 'r') as file:
            y = json.loads(file.read())

            # TODO: Noticed that, rarely, the 'sourceUrl' isn't formatted, need to create a catch for this
            url = y['sourceUrl']

            webbrowser.open(f'https://www.justtherecipe.com/?url={url}')


# Implementation of Python code for use in a Terminal emulator
if __name__ == '__main__':
    print('')
    print('Select an option: ')
    print('1) Select recipes for shopping')
    print('2) Output recipe to console')
    print('3) Open recipe card')

    choice = input()

    # Empty array that contains all of our recipes in the ~/Recipes folder
    allrecipes = []

    # Iterate and add to array
    for file in os.listdir(os.path.expanduser('~/Recipes/')):
        if file.endswith('.recipe'):
            allrecipes.append(file)

    # The user wants to go shopping!
    # Gather all the recipes the user wants from a comma-separated input, and process them with 'gather_ingredients' filter
    # Also outputs a Markdown file that contains a Markdown checklist with ingredients and measurements
    if int(choice) == 1:
        recipes = []

        print('')

        for file in allrecipes:
            print(f'{allrecipes.index(file) + 1}) {file}')

        print('')
        print('Input comma-separated index of the recipes')

        # Re-retrieve our input
        choice = input()

        # Separate input to get our comma-separated list
        for val in choice.strip().split(','):
            recipe = allrecipes[int(val.strip()) - 1]
            recipes.append(os.path.expanduser(f'~/Recipes/{recipe}'))

        # Get our consolidated ingredients list
        results = gather_ingredients(recipes)

        # Open our 'Shopping.md' file in write-create mode and write our output
        with open(os.path.expanduser(f'~/Recipes/Shopping.md'), 'w+') as file:
            for result in results:
                file.write(f'- [ ] {result}\n')

    elif int(choice) == 2:
        print('')
        for file in allrecipes:
            print(f'{allrecipes.index(file) + 1}) {file}')
        print('')
        print('Input the file index')
        index = int(input())
        recipe = allrecipes[index - 1]
        file = open(os.path.expanduser(f'~/Recipes/{recipe}'))
        read_recipe(file)
    elif int(choice) == 3:
        recipes = []
        print('')
        for file in allrecipes:
            print(f'{allrecipes.index(file) + 1}) {file}')
        print('')
        print('Input comma-separated index of the recipes')
        choice = input()
        for val in choice.split(','):
            recipe = allrecipes[int(val) - 1]
            recipes.append(os.path.expanduser(f'~/Recipes/{recipe}'))
        open_card(recipes)
