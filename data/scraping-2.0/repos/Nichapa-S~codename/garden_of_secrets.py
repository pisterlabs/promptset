import streamlit as st
import openai
import json
import pandas as pd


user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as a spy who gives codenames in English. You will receive a text and you will replace human names in the writing
        with the names of flowers. 
        If the human name is already a flower name, give that human a different flower name.
        Do not change the flower names that is not a human name in the context.
        The flower name you chose must relate to the human name it represents.
        If the human name in the context is a place or a brand, ignore it.
        Then list the codenames and human names that they mean in JSON array, one codename per line.
        each codename has 4 fields:
        - "name" - the human name that is detected
        - "codename" - the flower name you give
        - "reason" - reason why you chose that flower name
        - "examples of usage" - 1 sentence which has that human name in and replace it with codename. change the name only.
        """    


st.title(":ribbon: :violet[Garden of Secrets] :ribbon:")
st.markdown("Input your diary that you want to give some code names to. \n\
            The AI will suggest codenames for each name with names of flowers.\n\
            If your text has a specific name for a place, make sure to specify it. \n\
            ")

st.markdown("for examples : Harvard University, Green Gable Farm \n\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")

user_input = st.text_area("Enter some diary you want to keep it as a secret:", "⚘ ᠂ ⚘ Your text here  ⚘᠂ ⚘ ")

# submit button 
if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )

    # Show the response
    st.markdown('**AI response:**')
    suggestion_dictionary = response.choices[0].message.content

    str(suggestion_dictionary)

    sd = json.loads(suggestion_dictionary)

    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
    st.table(suggestion_df)