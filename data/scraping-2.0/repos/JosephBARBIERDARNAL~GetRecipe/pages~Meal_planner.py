import streamlit as st
import openai
import base64

from api import api_gpt
from front import make_space, csv_string_to_df, count_words
from ingredients import ingredients


st.title("Meal Planner")
st.markdown("""From a list of ingredients, you can generate all your dishes for the week, from Monday
            to Friday, lunch and dinner. You can enter your own ingredients or use pre-defined ones.
            You must at least enter 10 ingredients.""")
make_space(3)

#define gpt parameters
openai.api_key = st.secrets["openai_key"]
system_msg_summary = """You're an AI assistant who creates meal plans that run from Monday to Friday.
                        All the meals you propose must be extremely tasty. Your answers
                        must be in a CSV format with the following columns: Day, Meal, Recipe name, Recipe description.
                        Your answers must only be composed of 2 meals per day: Lunch and Dinner.
                        Your answer are only the CSV file, no need to write anything else."""

#Get user input
number_of_ingredients = 0
use_predefined_ingredients = st.checkbox("Use pre-defined ingredients", value=False)
if not use_predefined_ingredients:
    prompt = st.text_area("Enter your preferences")
    number_of_ingredients = count_words(prompt)
else:
    ingredients_list = ingredients()
    prompt = st.multiselect("Select your ingredients", ingredients_list, max_selections=50)
    prompt = [string[0].upper() + string[1:] for string in prompt]
    number_of_ingredients = len(prompt)
    prompt = ", ".join(prompt)
    st.write(f"You have {number_of_ingredients} ingredients\n\n {prompt}")

#call gpt api and display output
output_gpt = None
run = st.button("Find a meal plan!")
if run and number_of_ingredients > 10:
    with st.spinner("Loading"):
        output_gpt = api_gpt(prompt, system_msg_summary)
        make_space(1)

        #download csv
        df = csv_string_to_df(output_gpt)
        st.table(df)
        csv = output_gpt.encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="meal_plan.csv">Download my meal plan</a>'
        st.markdown(href, unsafe_allow_html=True)








# user ingredient suggestion by mail
make_space(15)
st.markdown("An ingredient is missing? Just start a [github issue](https://github.com/JosephBARBIERDARNAL/GetRecipe/issues)")
