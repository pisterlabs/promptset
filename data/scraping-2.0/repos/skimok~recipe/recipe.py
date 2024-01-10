from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Securely fetch the API key using the dotenv library
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
openai.api_key = key

dietary_restrictions = [
    "Gluten-Free",
    "Dairy-Free",
    "Vegan",
    "Pescatarian",
    "Nut-Free",
    "Kosher",
    "Halal",
    "Low-Carb",
    "Organic",
    "Locally Sourced",
]

cuisines = [
    "",
    "Italian",
    "Mexican",
    "Chinese",
    "Indian",
    "Japanese",
    "Thai",
    "French",
    "Mediterranean",
    "American",
    "Greek",
]


@app.route('/')
def index():
    # Display the main ingredient input page
    return render_template('index.html', cuisines=cuisines, dietary_restrictions=dietary_restrictions)


@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    # Extract the three ingredients from the user's input
    ingredients = request.form.getlist('ingredient')

    # Extract cuisine and restrictions
    selected_cuisine = request.form.get('cuisine')
    selected_restrictions = request.form.getlist('restrictions')

    print('selected_cuisine: ' + selected_cuisine)
    print('selected_restrictions: ' + str(selected_restrictions))

    if len(ingredients) != 3:
        return "Kindly provide exactly 3 ingredients."

    # Craft a conversational prompt for ChatGPT, specifying our needs
    # prompt = f"Craft a recipe in HTML using \
    #     {', '.join(ingredients)}. It's okay to use some other necessary ingredients. \
    #     Ensure the recipe ingredients appear at the top, \
    #     followed by the step-by-step instructions."
    prompt = f"{', '.join(ingredients)}을(를) 사용하여 HTML에서 레시피를 한국어로 만드세요. \
        다른 필요한 재료를 사용하는 것도 괜찮습니다. 레시피의 재료가 위에 나타나도록 하고, \
        그 다음에 단계별 지침을 따르세요."

    if selected_cuisine:
        prompt += f" The cuisine should be {selected_cuisine}."

    if selected_restrictions and len(selected_restrictions) > 0:
        prompt += f" The recipe should have the following restrictions: {', '.join(selected_restrictions)}."

    print('prompt: ' + prompt)

    messages = [{'role': 'user', 'content': prompt}]

    # Engage ChatGPT to receive the desired recipe
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )

    # Extract the recipe from ChatGPT's response
    recipe = response["choices"][0]["message"]["content"]

    # Showcase the recipe on a new page
    return render_template('recipe.html', recipe=recipe)


if __name__ == '__main__':
    app.run(debug=True)
