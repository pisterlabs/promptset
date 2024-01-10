import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openai
import os

# Initialize OpenAI API
openai.api_key = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate image prompt based on the statement
def generate_image_prompt(statement):
    prompt = f"A one sentence description of a generic family friendly movie scene with this topic: {statement}"
    image_prompt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    image_prompt = image_prompt_response['choices'][0]['message']['content'].strip()
    return image_prompt

# Function to generate counterarguments based on the statement
def generate_counterarguments(statement):
    prompt = f"Create a list of the top 3 counterarguments to the statement, rank them by the arguments logical strength (give the argument a score of 1-100 subtracting points for logical fallacies and cognitive distortions in the arguments, annotate the fallacies and distortions found), : {statement}"
    counter_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    counterarguments = counter_response['choices'][0]['message']['content'].strip().split('\n')
    return counterarguments

# Title
st.title("Galileo's Nuance finder")

# Dynamic Textbox for Surveyor
statement = st.text_input('Enter your Statement here:', '')

# Generate counterarguments, image prompt, and AI-created image
if statement:
    counterarguments = generate_counterarguments(statement)
    st.title("Here are some counterarguments:")
    for counterargument in counterarguments:
        st.write(counterargument)
    
    image_prompt = generate_image_prompt(statement)
    image_response = openai.Image.create(
        prompt=image_prompt,
        n=1,
        size="1024x1024"
    )
    image_url = image_response['data'][0]['url']
    st.image(image_url, caption=f"AI-generated image related to the topic: {statement}")

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

