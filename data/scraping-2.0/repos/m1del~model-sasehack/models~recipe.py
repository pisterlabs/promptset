import openai
import os
import readline
from dotenv import load_dotenv
load_dotenv()
import firebase_admin
from firebase_admin import credentials, firestore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIREBASE_JSON_PATH = os.path.join(BASE_DIR, 'firebase.json')

# Firebase Setup
cred = firebase_admin.credentials.Certificate(FIREBASE_JSON_PATH)
firebase_admin.initialize_app(cred)

db = firestore.client()

# Firebase Functions
def get_ingredients_to_expire():
    ingredients_with_expiration = []  # Will store tuples (ingredient, daysTillExpire)
    inventory = []
    ingredients_ref = db.collection(u'food')
    docs = ingredients_ref.stream()
    
    for doc in docs:
        data = doc.to_dict()
        expiration_days = int(data['daysTillExpire'])
        
        if expiration_days <= 3:
            ingredients_with_expiration.append((data['name'], expiration_days))
        else:
            inventory.append(data['name'])
    
    return ingredients_with_expiration, inventory
# def get_ingredients_to_expire():
#     ingredients = []
#     inventory = []
#     ingredients_ref = db.collection(u'food')
#     docs = ingredients_ref.stream()
#     for doc in docs:
#         data = doc.to_dict()
#         expiration_days = int(data['daysTillExpire'])
#         if expiration_days <= 3:
#             ingredients.append(data['name'])
#         else:
#             inventory.append(data['name'])
#     return ingredients, inventory


# OpenAI config
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

def get_recipe(ingredients, inventory):
    prompt_text = (f"I have most cookware available at my disposal such as a stove, oven, microwave, air fryer, toaster, kettle, along with utensils like kitchen knives, spatulas, etc. "
                   f"I have these ingredients: {ingredients}. "
                   f"Give me a recipe that includes most, if not all of these ingredients. "
                   f"You may include other ingredients from my kitchen, such as {inventory} "
                   "and items that are common to most households, like salt, pepper, oregano, and garlic powder. "
                   "What can I make as well as the instructions?")
    
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message['content']



def main():
    ingredients, inventory = get_ingredients_to_expire()
    print(ingredients)
    print(inventory)
    # recipe = get_recipe(ingredients, inventory)
    # print("\nRecipe and Instructions:\n", recipe)

if __name__ == "__main__":
    main()
