import gradio as gr 
import openai
import random

openai.api_key = "enter your key"


recipe_websites = [
    "https://www.indianhealthyrecipes.com/",
    "https://hebbarskitchen.com/",
    "https://www.harighotra.co.uk/indian-recipes",
    "https://www.vegrecipesofindia.com/recipes/pasta/",
    "https://www.vegrecipesofindia.com/recipes/indian-chinese/",
    "https://www.indianhealthyrecipes.com/paneer-recipes/"
]

def generate_recipe(time_available, num_people, diet_preference, experience_level, ingredients):
    prompt = f"Generate a recipe with the following parameters:\nTime Available: {time_available} minutes\nNumber of People: {num_people}\nDiet Preference: {diet_preference}\nExperience Level: {experience_level}\nAvailable Ingredients: {ingredients}\nRecipe:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    recipe = response.choices[0].text.strip()

    random_link = random.choice(recipe_websites)

    recipe_with_link = f"{recipe}\n\nYou can find more recipes like this at: {random_link}"
    return recipe_with_link

def generate_with_button(time_available, num_people, diet_preference, experience_level, ingredients):
    if ingredients is None:
        ingredients = ""
    recipe = generate_recipe(time_available, num_people, diet_preference, experience_level, ingredients)
    return recipe

inputs = [
    gr.inputs.Slider(minimum=10, maximum=240, step=10, label="Time Available (minutes)"),
    gr.inputs.Number(label="Number of People"),
    gr.inputs.Radio(["Vegan", "Low Calorie", "Regular"], label="Diet Preference"),
    gr.inputs.Radio(["Beginner", "Intermediate", "Advanced"], label="Experience Level"),
    gr.inputs.Textbox(label="Available Ingredients")
]


html = (
    "<div >"
    + "<img  src='file/recipe.png'  alt='image One'>"
    + "</div>"
)

gr.Interface(
    fn=generate_with_button,
    inputs=inputs,
    outputs=gr.outputs.Textbox(),
    live=False,
    title="Recipe Chatbot",
    # theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink),
    description=html,
).launch()  
