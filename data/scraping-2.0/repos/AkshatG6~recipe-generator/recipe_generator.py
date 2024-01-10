import os 
from apikey import apikey
import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# os.environ['OPENAI_API_KEY'] = apikey
# os.environ['OPENAI_API_KEY'] = apikey

os.environ['OPENAI_API_KEY'] = os.getenv('MY_API_KEY')

st.title('Recipe Generator')

preferredIngrediants = st.text_input('Enter key ingredients (do not leave empty)*:', help="Eg. Chicken, Rice, Tomato, etc.", placeholder= "Eg. Chicken, Rice, Tomato, etc.", value="")
calorie_limit = st.number_input('Calorie Limit (in cals)', min_value=0, max_value=5000, value=400)

alergic_ingrediants = st.text_input('Enter any alergic ingrediants:', help = "Eg. Nuts, Dairy, etc.", placeholder= "Eg. None or Nuts, Dairy, etc", value="")
meal_type = st.selectbox('Meal Type', ('Select a meal type...', 'Breakfast', 'Lunch', 'Dinner', 'Snack', 'Dessert'), index=0)

cooking_time = st.number_input('Cooking Time (in minutes)', min_value=0, max_value=200, value=30)
serving_size = st.number_input('Serving Size', min_value=0, max_value=8, value=1)
dietary_preference = st.selectbox('Dietary Preference', ('None', 'Vegan', 'Vegetarian', 'Ketogenic', 'Paleo', 'Primal', 'Whole30'), index=0)
nuitrition_needs = st.selectbox('Nuitrition Needs', ('None', 'Balanced', 'High-Protein', 'Low-Fat', 'Low-Carb'), index=0)
cuisin_type = st.selectbox('Meal Type', ('None', 'Italian', 'Chinese', 'Indian', 'Continental'), index=0)
skill_level = st.selectbox('Skill Level', ('Beginner', 'Intermediate', 'Advanced'), index=0)



# our prompt that will be sent to LLM
recipe_template = PromptTemplate(

    input_variables=['recipe_characterisitcs'],
    template=  """
               Give me a recipe with following characteristics.
               
               Preferred Ingredients: {recipe_characterisitcs}

               If no recipe is found, return "Could not generate any recipe. Please change some constrains". If recipes are found then send them in following template - "Recipe Name : ", "Recipe Ingredients : ", "Recipe Instructions : " Follow this template only and send the response.
            """
)

# we are limiting maximum tokens to 180 to control cost
llm = OpenAI(temperature=0.9, max_tokens=500)

# single chain as of now. 
# We can keep the history of our calls. Not used yet. 
recipe_chain = LLMChain(llm = llm, prompt=recipe_template, verbose=True, memory=ConversationBufferMemory(max_len=800))
  
# create a string to pass to the chain from the above variables
preferredIngrediants = 'Preferred Ingredients: ' + preferredIngrediants + '\n' + 'Calorie Range: ' + str(calorie_limit) + '\n' + 'Allergic Ingredients: ' + alergic_ingrediants + '\n' + 'Meal Type: ' + meal_type + '\n' + 'Cooking Time: ' + str(cooking_time) + '\n' + 'Dietary Preference: ' + dietary_preference + '\n' + 'Serving Size: ' + str(serving_size) + '\n' + 'Nutrition Needs: ' + nuitrition_needs + '\n' + 'Cuisine Type: ' + cuisin_type + '\n' + 'Skill Level: ' + skill_level + '\n'


## add a generate button to the streamlit app on clicking which we can call the recipe_chain.run() function
button = st.button('Generate Recipe')


if button:
    print(preferredIngrediants)
    with st.spinner('Generating the recipe...'):
        response = recipe_chain.run(recipe_characterisitcs = preferredIngrediants)
    # response = recipe_chain.run(recipe_characterisitcs = preferredIngrediants)
    # st.write(response)

    # print(response) # returned as string 

    # split response string into lines
    response_lines = response.split('\n')

    recipe_dict = {}

    current_key = ""
    for line in response_lines:
        if line.strip() == "":
            continue
        # check if line is a title (followed by ':')
        if ':' in line:
            current_key, value = line.split(':')
            current_key = current_key.strip()
            value = value.strip()
            recipe_dict[current_key] = value
        else:
            # if not a title, it is a continuation of a list
            recipe_dict[current_key] += "\n" + line.strip()


    st.title(recipe_dict['Recipe Name'])
    st.subheader('Ingredients')
    st.write(recipe_dict['Recipe Ingredients'])
    st.subheader('Instructions')
    st.write(recipe_dict['Recipe Instructions'])
    
    # print(recipe_dict)

    # print(recipe_dict['Recipe Name'])
    # print(recipe_dict['Recipe Ingredients'])
    # print(recipe_dict['Recipe Instructions'])
