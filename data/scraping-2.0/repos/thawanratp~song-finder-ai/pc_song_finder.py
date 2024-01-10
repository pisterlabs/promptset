import streamlit as st
import openai
import json
import pandas as pd
import re
import base64

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as an AI suggesting an English song for a user. You will receive a 
            description of a song and you should suggest a name of a song that matches a user's description.
            You should also describe the song by summarizing the full lyrics of the song in one long sentence.
            You should also list several words that relates to the lyrics of the song that you suggested.
            Remember to reply in the following format only:
            "The song is called 'name of the song' and it is by 'name of the artist'.
            The song is about 'description of the song'.
            Here are some interesting words to get you in the mood:" 
            Then you have to list several words.
            The words could be adjectives, nouns, verbs, adverbs, etc.
            Remember that the words reflects the lyrics of the song that you suggested.
            The list should immediately follow the format above.
            List words in a JSON array.
            Each word should have 4 fields:
            - "word" - the word reflecting the lyrics of the song that you suggested
            - "category" - the category of the word that you suggested (adjective, noun, verb, adverb, etc.)
            - "meaning" - the meaning of the word that you suggested
            - "example" - a sentence that shows how the word is used
            - "synonyms" - the synonyms of the word (give at least 3 synonyms)
            Remember that the word list must be in a JSON array.
            Don't say anything at first. Wait for the user to say something.
        """

st.title(':musical_score: Song Finder :mag:')
st.markdown('Creating a playlist? The AI will suggest you a song that matches your mood and give you some keywords to help you get the vibes.')

user_input = st.text_area("Describe a song you need to hear at the moment:", "Tell me you mood, your fav genre, or a song you like.")


# submit button after text input
if st.button('Search'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': user_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    # Show the response from the AI in a box
    st.markdown('**AI response:**')
    response_dictionary = response.choices[0].message.content
    
    text = response_dictionary
    pattern = r'^(The).*\s.*\s.*(mood:)'
    match = re.search(pattern, text)
    text_part = match.group(0)
    print(text_part)
    st.markdown(text_part)

    text = response_dictionary
    pattern = r'^(The).*\s.*\s.*(mood:)'
    replace = r''
    new_text = re.sub(pattern, replace, text)

    sd = json.loads(new_text)
    print(sd)

    words_df = pd.DataFrame.from_dict(sd)
    print(words_df)
    st.table(words_df)

    st.markdown("#### Get Your Keywords")
    csv = words_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="song_keywords_output.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


