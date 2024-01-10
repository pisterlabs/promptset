import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 


st.set_page_config(
    page_title="Diet Plan",
    page_icon="🥗📅",
)

st.markdown("<h2 style='color: green; font-style: italic; font-family: Comic Sans MS; ' >Healthy Wealthy Daily Diet Planner🥗📅</h2> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS; font-size:2rem'>Eat Well🥦, Live Wealthy 🌟: Your Personalized Diet Planner for Optimal Health!💰🍎</h3>", unsafe_allow_html=True)


prompt_diet = st.text_input("Input your diet preferences like the calories, cuisine type etc")

prompt_snacks = st.text_input("Craving for healthy snacks, enter your choice here to get healthy options.")

diet_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Consider {topic}, and give a diet plan for the entire day seperately for breakfast lunch and dinner.'
)

snacks_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Consider {topic}, and give what can I eat for snack time.'
)

# Llms
llm = OpenAI(temperature=0.9) 


diet_chain = LLMChain(llm=llm, prompt=diet_template, output_key='health')
snacks_chain = LLMChain(llm=llm, prompt=snacks_template, output_key='snacks')



# Show stuff to the screen if there's a prompt
if prompt_diet: 
    diet = diet_chain.run(prompt_diet)

    st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >Your diet plan for the day </p>", unsafe_allow_html=True)
    st.write(diet)

if prompt_snacks: 
    diet = diet_chain.run(prompt_snacks)

    st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >Healthy snack options</p>", unsafe_allow_html=True)
    st.write(diet)


 