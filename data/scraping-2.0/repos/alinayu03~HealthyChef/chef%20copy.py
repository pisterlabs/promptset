from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import os
import streamlit as st

# Title
st.markdown("# Healthy Chef 🧑‍🍳🍴")
st.markdown("""###
What you eat affects how you feel, and eating healthy and enough helps your body function at its best, especially when you have cancer. Proper nutrition is a key part of your cancer treatment and recovery. During treatment, you might need more calories and protein to help your body maintain weight, heal as quickly as possible, and fight infection. 

We know that eating well can be challenging when you have cancer as it may become difficult to follow your usual diet, you might develop side effects that affect your appetite once you start treatment, or it might become difficult financially to access healthy groceries. The Food to Overcome Outcomes Disparities (FOOD) Program through the Immigrant Health and Cancer Disparities Service at Memorial Sloan Kettering aims to provide you with the nutritional support to guide you during your cancer journey and reduce nutrition gaps among the medically underserved.

Unsure about what to cook next? This site features a recipe generator. Below, you can input ingredients you receive from our food pantries to get inspired for your next meal. 
""")

st.divider()

# Secret OpenAI API Key
openai_api_key = st.secrets["openai_api_key"]

# User input OpenAI API Key
# openai_api_key = col2.text_input("OpenAI API Key", type="password")

# Set key
os.environ["OPENAI_API_KEY"] = openai_api_key

# init options
ingredients = ""
meal_type = ""
culture = ""
high_protein = ""
low_carb = ""
sugar_free = ""
low_fat = ""
low_sodium = ""

col1, col2 = st.columns(2)

# Optional Preferences
culture = col2.text_input("Culture")
meal_type = col2.radio("Meal Type", ["Any", "Breakfast", "Lunch", "Dinner", "Snack"])
high_protein = col2.checkbox("High-protein")
low_carb = col2.checkbox("Low-carb")
sugar_free = col2.checkbox("Sugar-free")
low_fat = col2.checkbox("Low-fat")
low_sodium = col2.checkbox("Low-sodium")

# Ingredients input
ingredients = col1.text_area("Ingredients")

# LLM setup
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, temperature=0.0)

# Recipe Generator 
st.markdown("#### New Recipe")
template = """
        Task: Generate A Healthy Recipe with Nutrition Facts based on a list of ingredients and optional preferences
        Ingredient List: {ingredients}

        Optional Preferences:
        Meal Type: {meal_type}
        Culture: {culture}
        Dietary Restrictions:
        - High-protein: {high_protein}
        - Low-carb: {low_carb}
        - Sugar-free: {sugar_free}
        - Low-fat: {low_fat}
        - Low-sodium: {low_sodium}"""

# Recipe Generator Button
if st.button("Run", key="prompt_chain_button"):
    with st.spinner("Running"):
        input_variables = ["ingredients", "meal_type", "culture",
                           "high_protein", "low_carb", "sugar_free", "low_fat", "low_sodium"]
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template,
        )
        variables = {
            "ingredients": ingredients,
            "meal_type": str(meal_type),
            "culture": culture,
            "high_protein": high_protein,
            "low_carb": low_carb,
            "sugar_free": sugar_free,
            "low_fat": low_fat,
            "low_sodium": low_sodium,
        }
        chain = LLMChain(llm=llm, prompt=prompt)
        output = chain.run(variables)
        st.info(output)

# Nutrition Search
st.markdown("## Nutrition Search")

# Read CSV
df = pd.read_csv('nutrition.csv')

# Search
search_query = st.text_input("Enter a search query")
column_to_search = st.selectbox("Select a column to search", df.columns)

# Nutrition Search Button
if st.button("Search"):
    filtered_rows = df[df[column_to_search].str.contains(
        search_query, case=False)]
    st.write(filtered_rows)

# Display dataset
st.dataframe(df)


# Recommended daily macros

st.markdown("## Daily Recommended Macros")
image = "macros.jpg"
st.image(image, use_column_width=True)

# Nutrition Info
st.markdown("## Additional Resources")

st.write(
    """Here are some additional resources listed by topic. We hope that the following resources can empower you. However, this information is not intended to replace the advice of a medical professional. If you have any questions or concerns about your nutritional needs, you should talk to a doctor, nurse or dietitian. Be sure to talk to your cancer care team about any problems you’re having so they can help you manage them.

Managing Cancer Treatment Side-effects that affect nutrition: https://www.cancer.gov/about-cancer/treatment/side-effects/appetite-loss/nutrition-pdq#_177.

Information on nutrition: https://www.cancer.org/cancer/survivorship/coping/nutrition/benefits.html” 

Pamphlet from the American Cancer Society https://www.cancer.org/content/dam/cancer-org/cancer-control/en/booklets-flyers/nutrition-for-the-patient-with-cancer-during-treatment.pdf.

Information on reading nutrition facts: https://www.fda.gov/food/new-nutrition-facts-label/how-understand-and-use-nutrition-facts-label

Recipes for people with cancer: https://www.mskcc.org/experience/patient-support/nutrition-cancer/recipes#sort=relevancy&f:@marketing_recipe_symptoms=[Fatigue]

Your body needs a healthy diet to function at its best. This is even more important if you have cancer. 

""")
