import streamlit as st
import openai
import base64

from api import api_gpt
from front import make_space, csv_string_to_df
from ingredients import recipes

st.title("Grocery list generator")
st.markdown("""From a list of meal, you can generate the grocery list needed in order to cook them.""")
make_space(3)

#define gpt parameters
openai.api_key = st.secrets["openai_key"]
system_msg_summary = """You're an AI assistant who creates grocery lists based on meals.
                        Your answers must be in a CSV (comma separated value) format with the following columns: Ingredient, Quantity, Comment.
                        Your answer are only the CSV file, no need to write anything else."""

#Get user input
use_predefined_recipes = st.checkbox("Use pre-defined meals", value=False)
if not use_predefined_recipes:
    recipes = st.text_area("Enter the meals you want to cook")
    prompt = recipes
else:
    recipes = recipes()
    recipes = st.multiselect("Select the meals you want to cook", recipes, max_selections=10)
    prompt = [string[0].upper() + string[1:] for string in recipes]
    prompt = ", ".join(prompt)

#call gpt api and display output
output_gpt = None
run = st.button("Create grocery list!")
if run:
    make_space(1)
    for recipe in recipes:
        make_space(2)
        output_gpt = api_gpt(recipe, system_msg_summary)
        st.markdown("#### " + recipe)

        # download csv
        df = csv_string_to_df(output_gpt)
        st.table(df)
        csv = output_gpt.encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{recipe}.csv">Download the list</a>'
        st.markdown(href, unsafe_allow_html=True)








