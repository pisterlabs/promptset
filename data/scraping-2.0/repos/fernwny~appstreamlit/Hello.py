import streamlit as st
import openai
import json
import pandas as pd


# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")
client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as a lyricist and write a song about a topic of your choice. The song should be at least 2 verses long.The song should be about the topic you choose. The song should have at least 3 words that have the same vowel sound.
list the first verse to ninth and the vocabulary of the song in a JSON array without index.
-first line should be the first verse
-second line should be the second verse
-third line should be the third verse
-fourth line should be the fourth verse
-fifth line should be the chorus
-sixth line should be the sixth verse
-seventh line should be the seventh verse
-eighth line should be the eighth verse
-ninth line should be the ninth verse
-tenth line should be the vocabulary in the verses ,type and meaning
"""
st.title("MAI Lyricist")
st.markdown("This app uses the OpenAI API to generate lyrics based on topics of your choice.")
st.write("Enter a topic and click the button to generate lyrics. You can multiple topics to generate lyrics by separating them with a space.")
st.write("Example : hell heaven ")
# Get the topic from the user
topic = st.text_input("Topic", "ENTER TOPIC HERE")
if st.button('let\'s go'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': topic},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    # Show the lyrics to the user
    st.markdown('**Lyrics:**')
    suggestion_dictionary = response.choices[0].message.content
    sd = json.loads(suggestion_dictionary)
    if isinstance(sd, list):
        for i, verse in enumerate(sd[:-1], 1):  # Exclude the last item (vocab)
            if i == 5:
                st.write(f"Chorus: {verse}")
            elif i >= 6:
                st.write(f"Verse {i-1}: {verse}")
            else:
                st.write(f"Verse {i}: {verse}")
    else:
        st.error("The response is not a list.")
    
    
 
    print (sd)
    suggestion_df = pd.DataFrame.from_dict(sd)
    print(suggestion_df)
 
    # Show the vocabulary, type and meaning to the user

    st.markdown('**Vocabulary:**')
    vocab = sd[9]
    if isinstance(vocab, dict):
        for i, (word, meaning) in enumerate(vocab.items(), 1):
            st.write(f"{i}. {word} - {meaning}")
    elif isinstance(vocab, list):
        vocab_meanings = {word_info['word']: word_info['meaning'] for word_info in vocab}
        st.write(pd.DataFrame.from_dict(vocab_meanings, orient='index', columns=['Meaning']))
