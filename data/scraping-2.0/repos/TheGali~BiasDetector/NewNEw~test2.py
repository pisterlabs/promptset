import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openai
import os
import random
import re

# Initialize OpenAI API
# Note: Use environment variables for API keys for security reasons.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate counterarguments based on the statement
def generate_counterarguments(statement):
    prompt = f"List the top 3 arguments that someone might use to prove the statement, rank them by the arguments strength (give the argument a Power-Score of 1-100 subtracting points for logical fallacies and cognitive distortions in the arguments, annotate the fallacies and cognitive distortions found, list the subtractions), steel man everything and assume the best of the opposing view : {statement}"
    counter_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    counterarguments_raw = counter_response['choices'][0]['message']['content'].strip()
    counterarguments_list = counterarguments_raw.split('\n\n')  # Assuming each counterargument is separated by two newlines
    return counterarguments_list[:3]  # Get only the top 3 counterarguments

# Function to generate random percentages
def generate_random_percentages(num):
    return [random.randint(1, 100) for _ in range(num)]

# Title
st.title("Galileo's Nuance finder")

# Dynamic Textbox for Surveyor
statement = st.text_input('Enter your Statement here:', '')

# Generate counterarguments and chart only when a statement is entered
if statement:
    counterarguments = generate_counterarguments(statement)
    agreement_percentages = generate_random_percentages(len(counterarguments))
    
    st.title("Here are some counterarguments:")
    
    for counterargument, percentage in zip(counterarguments, agreement_percentages):
        score_match = re.search(r'Score: (\d+)', counterargument)
        if score_match:
            logic_score = score_match.group(1)
            modified_counterargument = counterargument + f" (Agreement: {percentage}%)"
        else:
            modified_counterargument = f"{counterargument} (Agreement: {percentage}%)"
        
        st.write(modified_counterargument)
    
    # Generate random survey responses stratified by age
    total_responses = 100
    categories = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    colors = ['#ADD8E6', '#90EE90', '#FFFF00', '#FFA500', '#FF0000']

    stratified_responses = np.zeros((len(age_groups), len(categories)))
    for i in range(len(age_groups)):
        stratified_responses[i] = np.random.multinomial(total_responses // len(age_groups), [1/len(categories)]*len(categories))

    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom_values = np.zeros(len(categories))

    for i, age_group in enumerate(age_groups):
        ax.bar(categories, stratified_responses[i], bottom=bottom_values, label=f'Age {age_group}', color=colors[i], alpha=0.8)
        bottom_values += stratified_responses[i]

    ax.set_ylabel('Number of Responses')
    ax.set_xlabel('Response Categories')
    ax.set_title('Stratified Distribution of Answers by Age Group (Stacked)')
    ax.legend()
    ax.grid(axis='y')

    st.pyplot(fig)
