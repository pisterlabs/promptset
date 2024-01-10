import streamlit as st
import pandas as pd
import openai

openai.api_key = 'YOUR_API_KEY'

budget = st.number_input("Enter your budget (in Euros)", min_value=0)
age = st.number_input("Enter your age", min_value=0, max_value=120)
height = st.number_input("Enter your height (in cm)", min_value=0)
weight = st.number_input("Enter your weight (in kg)", min_value=0)
dietary_restrictions = st.text_input("Enter any dietary restrictions")

if st.button("Generate Diet Plan"):
    with st.spinner("Generating Diet Plan..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a dietitian. You provide a personalized diet plan for a {age}-year-old with a budget of {budget} euros, height {height} cm, weight {weight} kg, and dietary restrictions: {dietary_restrictions}. Just the weekly plan is enough.",
                },
                {
                    "role": "user",
                    "content": "What is the diet plan? Please provide the diet plan for a week in markdown format with a table. The table should include these columns: | Day | Breakfast | Lunch | Dinner | Snack | Cost (euros) | Calories |",
                },
            ],
        )

        print(response.choices[0].message["content"])
        st.write(
            "Your Personalized Diet Plan:\n", response.choices[0].message["content"]
        )
