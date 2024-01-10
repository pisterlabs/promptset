import openai
import streamlit as st
import json
import pandas as pd

user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as a translator. You will receive a Thai story 
            You should translate a Thai story to intermediate English.
            List the translation in a JSON array, one translation per line.
            Each translation should have 4 fields:
            - "Thai" - the Thai story
            - "English" - the English version of the story
            - "Vocabulary" - the vocabulary words you should know.
            - "CEFR" - the vocabulary words in vocabulary field with CEFR level.
            Make the story interesting and fun."""

st.title(f":christmas_tree::santa: :red[{'TH-EN'}] :green[{'Christmas'}] :red[{'and'}] :green[{'New Year'}] :red[{'story translator'}]:confetti_ball::tada:")
st.markdown(f"Input the Christmas or New year story that you want to translate. \n\
            The AI will translate and give you some vocabulary words which you should know .")
st.markdown('กรุณาใส่เรื่องราววันคริสต์มาสหรือปีใหม่ที่คุณต้องการแปล AI จะแปลพร้อมแสดงคำศัพท์ภาษาอังกฤษที่ควรรู้')

user_input = st.text_area("เขียนเรื่องราวที่ต้องการแปล:") 

if st.button('ส่ง :gift:'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    
    st.markdown('**AI response:**')
    result_dictionary = response.choices[0].message.content
    rd = json.loads(result_dictionary)
    result_df = pd.DataFrame.from_dict(rd)
    st.table(result_df)

    st.markdown(":heart: Merry Christmas and Happy New Year :green_heart:")
    st.markdown(":balloon: สุขสันต์วันคริสต์มาสและสวัสดีปีใหม่ :partying_face:")
    

    
   

