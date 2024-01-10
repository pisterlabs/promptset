import streamlit as st
import openai
import json
import pandas as pd


user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)

prompt = """Act as an AI Mandarin assistant. You will receive a 
            piece of writing and you should give word's information.
            List the informations in a JSON array.
            Each information should have 4 fields:
            - "word" - the word in a piece of writing that is interesting            
            - "pinyin" - pinyin of the word
            - "part of speech" - part of speech of the word - noun, verb, adjective, adverb, etc.
            - "difficulty" - difficulty level for each word - beginner, intermediate, advanced
            - "translation" - word's translation in English

            Don't say anything at first. Wait for the user to say something.
        """    


st.title('Mandarin detector')
st.markdown('Input a Mandarin paragraph. \n\
            The AI will give you the interesting information in your paragraph.')

userinput = st.text_area("Enter your paragraph:", "Your text here")



if st.button('Submit'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': userinput},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    
    st.markdown('**response:**')
    suggestion_dictionary = response.choices[0].message.content



    sdict = json.loads(suggestion_dictionary)

    print (sdict)
    response_df = pd.DataFrame.from_dict(sdict)
    print(response_df)
    st.table(response_df)
