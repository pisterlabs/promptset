import streamlit as st
import openai
import requests
import nltk
import string
import json
import pandas as pd
import json.decoder
import time
from openai import BadRequestError
from bs4 import BeautifulSoup

st.sidebar.title("Welcome to AO3 Fiction Vocabulary Analysis!")
user_api_key = st.sidebar.text_input("OpenAI API Key", type="password", label_visibility='visible')
client = openai.OpenAI(api_key=user_api_key)

st.header("AO3 Fiction Vocabulary :orange[Analysis]")
st.markdown('Enter your AO3 fiction link down below. The AI will give you a summary from the chapter and\n\
            a list of vocabulary with the meaning, part of speech, example and synonyms according to your choice of :red[CEFR] vocabulary level.')

user_input_url = st.text_input("Your AO3 link :", 'https://archiveofourown.org/works/52126093')

vocab_level = st.selectbox(
    'Vocabulary Level :',
    ('B1', 'B2', 'C1', 'C2', 'Random'), label_visibility='visible'
)


submit = st.button('Submit')

max_retries = 10
retry_count = 0

while submit and retry_count < max_retries:

    url = user_input_url
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f'Failed to get the page, status code: {response.status_code}')
        st.stop()

    full_article = ''
    article = ''  
    content = response.text
    soup = BeautifulSoup(content, 'html.parser')

    words_tags = soup.findAll('dd', class_='words')
    word_count = words_tags[0].text.strip()
    word_count = word_count.replace(',', '')

    name_tags = soup.findAll('h2', class_='title heading')
    fiction_name = name_tags[0].text.strip()

    author_tags = soup.findAll('h3', class_='byline heading')
    author = author_tags[0].text.strip()

    nltk.download('stopwords')
    nltk.download('punkt')

    div_tags = soup.findAll('div', class_='userstuff')
    for div_tag in div_tags:
        p_list = div_tag.findAll('p')
        for p in p_list:
            if p.text.strip():
                full_article += p.text.strip() + ' '
                # filter out stop words
                # from nltk.corpus import stopwords
                # nltk.download('stopwords')
                stop_words = set(nltk.corpus.stopwords.words('english'))
                # remove punctuation from text
                table = str.maketrans('', '', string.punctuation)
                # remove remaining tokens that are not alphabetic
                text = nltk.word_tokenize(p.text)
                stripped = [w.translate(table) for w in text]
                words = [word for word in stripped if word.isalpha()]
                words = [w for w in words if not w in stop_words]
                article += ' '.join(words) + ' '

    article = article.replace("\'", "\"")

    if len(article) > 11000: # 10780
        # st.error('Your fiction is too long for AI to analyze. Please select another fiction that has less than 3000 words.')
        st.warning(f"**{fiction_name}** by **{author}** has {len(article)} words.")
        st.error(f"**{fiction_name}** by **{author}** overall token length is longer than this model's maximum token length. Please try again with a another fiction less than 3000 words.")
        st.stop()
    
    else:
        # if user choose B1
        if vocab_level == 'B1':
            prompt = """Act as an AI English tutor who knows much about vocabulary. You will receive a fiction from AO3
                and you should give list of 15 interesting words that has the level of CEFR only B1 and the meaning of that word in Thai language.
                List the vocabulary in a JSON array.
                Each line in the JSON array should have these fields:
                - "Vocabulary" - the vocabulary that has CEFR level only B1
                - "CEFR" - the CEFR level of the word
                - "Part of Speech" - the part of speech of the word
                - "Meaning" - the meaning og the word in Thai language
                - "Example" - the example use of the word in English
                - "Synonym" - the synonyms of the word in English
                Make sure that the result is JSON array.
                Make sure that the JSON array is valid.
                Don't say anything except result that is JSON array
                Don't say anything at first. Wait for the user to say something.
            """ 
        # if user choose B2
        elif vocab_level == 'B2':
            prompt = """Act as an AI English tutor who knows much about vocabulary. You will receive a fiction from AO3
                and you should give list of 15 interesting words that has the level of CEFR only B2 and the meaning of that word in Thai language.
                List the vocabulary in a JSON array.
                Each line in the JSON array should have these fields:
                - "Vocabulary" - the vocabulary that has CEFR level only B2
                - "CEFR" - the CEFR level of the word
                - "Part of Speech" - the part of speech of the word
                - "Meaning" - the meaning og the word in Thai language
                - "Example" - the example use of the word in English
                - "Synonym" - the synonyms of the word in English
                Make sure that the result is JSON array.
                Make sure that the JSON array is valid.
                Don't say anything except result that is JSON array
                Don't say anything at first. Wait for the user to say something.
            """
        # if user choose C1
        elif vocab_level == 'C1':
            prompt = """Act as an AI English tutor who knows much about vocabulary. You will receive a fiction from AO3
                and you should give list of maximum 15 interesting words that has the level of CEFR only C1 and the meaning of that word in Thai language.
                List the vocabulary in a JSON array.
                Each line in the JSON array should have these fields:
                - "Vocabulary" - the vocabulary that has CEFR level only C1
                - "CEFR" - the CEFR level of the word
                - "Part of Speech" - the part of speech of the word
                - "Meaning" - the meaning og the word in Thai language
                - "Example" - the example use of the word in English
                - "Synonym" - the synonyms of the word in English
                Make sure that the result is JSON array.
                Make sure that the JSON array is valid.
                Don't say anything except result that is JSON array
                Don't say anything at first. Wait for the user to say something.
            """
        # if user choose C2
        elif vocab_level == 'C2':
            prompt = """Act as an AI English tutor who knows much about vocabulary. You will receive a fiction from AO3
                and you should give list of maximum 15 interesting words that has the level of CEFR only C2 and the meaning of that word in Thai language.
                List the vocabulary in a JSON array.
                Each line in the JSON array should have these fields:
                - "Vocabulary" - the vocabulary that has CEFR level only C2
                - "CEFR" - the CEFR level of the word
                - "Part of Speech" - the part of speech of the word
                - "Meaning" - the meaning og the word in Thai language
                - "Example" - the example use of the word in English
                - "Synonym" - the synonyms of the word in English
                Make sure that the result is JSON array.
                Make sure that the JSON array is valid.
                Don't say anything except result that is JSON array
                Don't say anything at first. Wait for the user to say something.
            """
        # if user choose All
        elif vocab_level == 'Random':
            prompt = """Act as an AI English tutor who knows much about vocabulary. You will receive a fiction from AO3
                and you should give list of 15 interesting words that has the level of CEFR from B1 to C2 and the meaning of that word in Thai language.
                List the vocabulary in a JSON array.
                Each line in the JSON array should have these fields:
                - "Vocabulary" - the vocabulary that has CEFR level from B1 to C2
                - "CEFR" - the CEFR level of the word
                - "Part of Speech" - the part of speech of the word
                - "Meaning" - the meaning og the word in Thai language
                - "Example" - the example use of the word in English
                - "Synonym" - the synonyms of the word in English
                Make sure that the result is JSON array.
                Make sure that the JSON array is valid.
                Don't say anything except result that is JSON array
                Don't say anything at first. Wait for the user to say something.
            """


        messages_so_far = [
            {"role": "system", "content": prompt},
            {'role': 'user', 'content': article},
        ]

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_so_far
        )


        prompt_summarize = """Act as an AI English native speaker who is a writer and likes to read fiction. You will receive a 
                fiction from AO3 and you should summarize that chapter you got and maximum 4 sentences.
                Don't say anything at first. Wait for the user to say something.
            """

        completion2 = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_summarize},
                {"role": "user", "content": full_article}
            ]
        )

        # subheading for the fiction name and author
        st.subheader(f"***{fiction_name}*** by {author}")
        # st.markdown('**AI response:**')

        # SUMMARY
        st.write("**Summary :**")
        summarized_chapter = completion2.choices[0].message.content
        # display the summarized chapter by AI
        st.write(summarized_chapter)

        # VOCABULARY
        st.write(f"**Vocabulary List :**")
        
            
        try:

            # Inner loop for retrying when suggestion_df is empty or doesn't have 15 rows
            max_retries_inner = 10  # Adjust the maximum inner retries as needed
            retry_count_inner = 0

            while True:
                vocabulary = completion.choices[0].message.content

                vd = json.loads(vocabulary)
                suggestion_df = pd.DataFrame(vd)

                if suggestion_df.empty or suggestion_df.shape[1] < 5:
                    retry_count_inner += 1

                    if retry_count_inner == max_retries_inner:
                        st.error(f"Inner retry limit ({max_retries_inner}) reached. Please submit again or change the fiction.")
                        break  # Break out of the inner loop if the maximum retries are reached

                    time.sleep(1)  # Adjust the delay time as needed

                else:
                    st.dataframe(suggestion_df)
                    break  # Break out of the inner loop if successful

            break  # Break out of the outer loop if successful

        except (json.decoder.JSONDecodeError, ValueError) as e:
                retry_count += 1

                if retry_count == max_retries:
                    st.error(f"Maximum retry limit ({max_retries}) reached. Please submit again or change the fiction.")

                time.sleep(1)  # Adjust the delay time as needed

        except BadRequestError:
            st.error(f"**{fiction_name}** by **{author}** overall length is longer than this model's maximum token length. Please try again with a another fiction.")
            break