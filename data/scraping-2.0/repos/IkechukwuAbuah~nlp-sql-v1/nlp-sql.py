# import librairies
import openai
import numpy as np
import streamlit as st

# Load OpenAI API key from secrets file
openai.api_key=st.secrets["pass"]


# Provide context of the app
#_______________Optionnal_______________________
st.sidebar.header('A simple Natural language to SQL query app. ')

st.sidebar.image("sql.png", use_column_width=True)

st.sidebar.write("""
         ######  We use streamlit and OpenAI APIs to translate natural 
            language queries into SQL statements. Input your query in natural language, and the app will generate the corresponding SQL statement for you to execute.
         """)
st.sidebar.write("""
         ######  Made by Kelvin I. Abuah  [LinkedIn](https://www.linkedin.com/in/ikechukwuabuah/), [Github](https://github.com/IkechukwuAbuah)
         """)
st.header('Natural language -> SQL queries.')
#___________________________________________________




# Text input where the user enter the text to be translated to SQL query
query= st.text_input('Enter you text to generate SQL query', '')

#The query is sent to the OpenAI API  throught the prompt variable using 
#the "text-davinci-002" engine, and the generated response is returned as 
#a string.
#These  parameters configuration where based on the ones provided by openai
def generate_sql(query):
    model_engine = "text-davinci-003"
    prompt = (
        f"Translate the following natural language query to SQL:\n"
        f"{query}\n"
        f"SQL:"
    )
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )
    return response.choices[0].text.strip()

#if the Generate SQL query is clicked 
if st.button('Generate SQL query'):
  #if text input is not empty
  if len(query) > 0:
    #Generate sql query
    Respo=generate_sql(query)
    #print sql query
    st.write(Respo)
