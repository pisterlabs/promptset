import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('☮️♻️ Gen-Z Health Minder 🏋🏻‍♂️🥗')

st.title("Employee Mental Health Data")
st.write("Enter the following information:")

# Llms
llm = OpenAI(temperature=0.9)



def get_user_input():
    age = st.number_input("Age", min_value=18, max_value=65, value=25)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    height = st.number_input("Height (in cm)", min_value=100, max_value=300, value=175)
    weight = st.number_input("Weight (in kg)", min_value=10, max_value=500, value=82)
    family_history = st.radio("Do you have a family history of mental illness?", ["Yes", "No"])
    work_interfere = st.radio("If you have a mental health condition, do you feel that it interferes with your work?", ["Often", "Rarely" , "Never" , "Sometimes"])
    remote_work = st.radio("Do you work remotely (outside of an office) at least 50% of the time?", ["Yes", "No"])
    benefits = st.radio("Does Employer Provide Mental Health Benefits?", ["Yes", "No" , "Dont_Know"])
    care_option = st.radio("Do you know the options for mental health care your employer provides?", ["Yes", "No" , "Dont_Know"])
    wellness_program = st.radio("Has your employer ever discussed mental health as part of an employee wellness program?", ["Yes", "No" , "Dont_Know"])
    seek_help = st.radio("Does your employer provide resources to learn more about mental health issues and how to seek help?", ["Yes", "No" , "Dont_Know"])
    anonymity = st.radio("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?", ["Yes", "No" , "Dont_Know"])
    mental_health_consequence = st.radio("Do you think that discussing a mental health issue with your employer would have negative consequences?", ["Yes", "No" , "Maybe"])
    physical_health_consequence = st.radio("Do you think that discussing a physical health issue with your employer would have negative consequences?", ["Yes", "No" , "Maybe"])
    coworkers = st.radio("Willingness to Discuss Mental Health with Coworkers?", ["Yes", "No"])
    supervisor = st.radio("Willingness to Discuss Mental Health with Supervisor?", ["Yes", "No"])
    mental_health_interview = st.radio("Would you bring up a mental health issue with a potential employer in an interview?", ["Yes", "No" , "Maybe"])
    physical_health_interview = st.radio("Would you bring up a physical health issue with a potential employer in an interview?", ["Yes", "No" , "Maybe"])
    mental_vs_physical = st.radio("Do you feel that your employer takes mental health as seriously as physical health?", ["Yes", "No" , "Dont Know"])
    obs_consequence = st.radio("Observed Negative Consequences for Coworkers with Mental Health Conditions?",
                               ["Yes", "No"])

    # Return the user input as a dictionary
    user_input = {
        "Age": age,
        "Gender": gender,
        "Height": height,
        "Weight": weight,
        "Family History": family_history,
        "Work Interfere": work_interfere,
        "Remote Work": remote_work,
        "Benefits": benefits,
        "Care Option": care_option,
        "Wellness Program": wellness_program,
        "Seek Help": seek_help,
        "Anonymity": anonymity,
        "Mental Health Consequence": mental_health_consequence,
        "Physical Health Consequence": physical_health_consequence,
        "Coworkers": coworkers,
        "Supervisor": supervisor,
        "Mental Health Interview": mental_health_interview,
        "Physical Health Interview": physical_health_interview,
        "Mental Vs Physical": mental_vs_physical,
        "Obs_consequence": obs_consequence
    }

    user_input_string = str(user_input)  # Convert user_input dictionary to string
    user_input_string += "\n\nAbove is employee data working in the organization.\nPlease calculate the BMI index and provide a week-wise diet plan, physical exercise plan, and mental health exercise plan for a month with daily 60 minutes."
    #st.write(user_input_string)
    return user_input


user_input = get_user_input()
generated_promp =""
#Display the collected user input
#st.subheader("User Input:")
for key, value in user_input.items():
    #st.write(f"{key}: {value}")
    generated_promp += f"{key}: {value}"
    generated_promp += "\n"

generated_promp += "above is employee data working in organization\nplease calculate bmi index & provide week wise diet & physical & mental health exercise plan for a month for daily 60 min"
print(generated_promp)
if st.button("Submit"):
    # Generate the response here
    st.subheader("Response:")
    #st.write(generated_promp)
    response=llm(generated_promp)
    st.write(response)
