import streamlit as st
import openai
import json
import pandas as pd


user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

openai.api_key = user_api_key
prompt = """You will receive an English input from the user. Your task is to make the input easier to understand. 
For example, replace hard words with their simpler synonyms and put brackets over the new word, or rephrase the sentence to make it easier to understand.
If the word is, according to the CEFR level, B2 or upper, replace it with a word that is B1 or lower. 
Also, provide explanations or definitions for those hard-to-understand words in separate paragraphs. 
For instance, if the original word is 'complicated', you can replace it with 'complex' and provide a definition for 'complex' in the following paragraph. 
Similarly, if the word is 'societal', you can replace it with '[social]' and provide a definition for '[social]' in the following paragraph.
Also give each word an example sentence in that same paragraph.

# Instructions for definitions:
- For each replaced word, include a definition in a separate paragraph.
- Start each definition with the replaced word in square brackets, followed by the definition.

How the answer should look like:

if the input is:
'The societal impact of the pandemic is complicated.',

the output should be:
'The social impact of the pandemic is complex.

# Definitions:
[societal] Relating to society or social relations. (Example: 'Like cooking, cleaning, shopping, and washing, men can do it too, but the societal perception is that these are women's tasks or feminine gendered activities.')
[complicated] Involving many different and confusing aspects. (Example: 'The situation is complicated by the fact that many teachers are unwilling to return to work.')'
"""

st.title('Easier English')
st.markdown('Input a piece of text with challenging vocabulary, let the AI help simplify it. We will also provide definitions of the parts that might seem hard to comprehend.')

user_input = st.text_area("Enter text to simplify:", "Your text here")


# if st.button('Send'):
#     messages_so_far = [
#         {"role": "system", "content": prompt},
#         {'role': 'user', 'content': user_input},
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages_so_far,
#         max_tokens=1000
#     )

#     ai_response = response['choices'][0]['message']['content']

#     simplified_text, definitions_part = ai_response.split('Definitions:', 1)

#     st.markdown('**Simplified Text:**')
#     st.write(simplified_text.strip())

#     st.markdown('**Definitions:**')
#     for line in definitions_part.splitlines():
#         if line.startswith('['):
#             word, definition = line.split('] ', 1)
#             st.markdown(f"- **{word}]** {definition.strip()}")

# if st.button('Send'):
#     messages_so_far = [
#         {"role": "system", "content": prompt},
#         {'role': 'user', 'content': user_input},
#     ]
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages_so_far,
#         max_tokens=700
#     )

#     ai_response = response['choices'][0]['message']['content']

#     st.markdown('**Simplified Text:**')
#     if 'Definitions:' in ai_response:
#         simplified_text, definitions_part = ai_response.split('Definitions:', 1)

#         st.write(simplified_text.strip())

#         st.write(definitions_part.strip())

#     else:
#         st.write(ai_response.strip())

if st.button('Send'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far,
        max_tokens=1000
    )

    ai_response = response['choices'][0]['message']['content']

    if 'Definitions:' in ai_response:
        simplified_text, definitions_part = ai_response.split('Definitions:', 1)

        st.markdown('**Simplified Text:**')
        st.write(simplified_text.strip())

        st.markdown('**Definitions:**')
        for line in definitions_part.splitlines():
            if line.startswith('['):
                word, definition = line.split('] ', 1)
                st.markdown(f"- **{word}]** {definition.strip()}")
    else:
        st.write(ai_response.strip())