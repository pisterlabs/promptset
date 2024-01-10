import streamlit as st
import openai
import time
from streamlit_extras.streaming_write import write

# OpenAI API key setup start----


addapi = st.text_input("Add your API Key below:", key="api")

api_key = addapi

if api_key is None:
    raise Exception("API key not found in .env file")

# OpenAI API key setup finish----

# -----------------------------------
   
openai.api_key = api_key

# Function to communicate with ChatGPT
def chat_with_gpt3_5(prompt):
    openai.api_key = api_key

    gpt3_5response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
        "role": "system",
        "content": "You are my personal assistant. You have all of the capabilities of GPT 4 and have an iq of over 200. Answer all questions to the best of your ability."
            },
            {
        "role": "user",
        "content": prompt
            }
            ],  
        max_tokens=1000  # Adjust the max_tokens as needed
    )
            

    return gpt3_5response['choices'][0]['message']['content']

def chat_with_gpt3(prompt):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-babbage-001",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None
    )

    return response.choices[0].text.strip()

#  ------------------------------------------------------------


mtitle = ('## GPT-4!')
def gpt_interface():
    for word in mtitle.split():
        yield word + " "
        time.sleep(0.23)  
write(gpt_interface)
 

with st.expander("Important:"):
        st.write('''
           - As of now, once you leave the page, your answers will disappear. Make sure you take a picture
                 or copy the answers that you need. 
           - If you dont see "Please wait, your answers will load shortly......" after pressing "Generate" press again. 
        ''')
 

prompt = st.text_input("")

# stoggle(
# "Instructions:",
# """
# For Ingredients mode: Provide what ingredients you would like a recipe with in the text field.
# For the Dish mode: Provide a name of a dish you would like to make MealMaker will come up with a recipe for you. If you have any allergies, dont forget to put them in the allergy text field. Then click 'Cook me a meal!' to generate your recipe.
# """,
# ) 
# st.divider()         

# Send query to the chatbot
if st.button("Generate"):
    pbtext = ("Please wait, your answers will load shortly......")
    prg = st.progress(0, text=pbtext)

    response = chat_with_gpt3_5(prompt)
            
    # Split the response into lines and find the recipe name
    lines = response.split('\n')
    name = ""
    steps = ""
    for line in lines:
        if "recipe name" in line.lower():
            recipe_name = line.strip()
        else:
            steps += line.strip() + "\n"
            
    for i in range(99):
        time.sleep(0.2)
        prg.progress(i+1, text=pbtext)
    
    for i in range(1):
        time.sleep(0.2)
        prg.progress(100, text="All done!")

        # Output
    with st.container():
        st.markdown(f"## {name}")
        st.markdown(steps)

    #  ----------------------------------------

