import streamlit as st
import openai
import json
import pandas as pd
# Get the OpenAI API key from the environment variable
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")
client = openai.OpenAI(api_key=user_api_key)
prompt = """Act as an assistant to help users paraphrase their sentences for better understanding. 
            You will receive sentences in the format [User's sentence]. 
            You have 2 tasks first is to generate a new sentence using simpler language based on the chosen language level ,second is you must return at least 3 words that are above the chosen language level.
            **Example:**
            Sentence: The implications of quantum entanglement on the measurement problem in quantum mechanics have been a subject of intense debate among physicists.
            Language Level: High School

            **Task:**
            Paraphrase the sentence using simpler language based on the chosen language level and find words that are above the chosen language level. For example, if the chosen language level is High School, the vocabulary list should be at least at the University level.
            If the chosen language level is University, the vocabulary list should be at least at the University level.
            Return the following information in JSON format:
            ```json
            {
            "original_text": "The original sentence",
            "paraphrased_text": "The paraphrased sentence",
            "vocabulary_list": [
                {
                "original_word": "Vocabulary1",
                "synonyms": ["Synonym1", "Synonym2", "Synonym3"],
                "example": "A sample sentence using a synonym"
                },
                {
                "original_word": "Vocabulary2",
                "synonyms": ["SynonymA", "SynonymB", "SynonymC"],
                "example": "A sample sentence using a synonym"
                },
                ...
            ]
            }
            """
st.title('Easy-Reading')
st.markdown('Input the complex sentence.\n'
            'The AI will paraphrase the sentence based on the chosen language level .')
sentence_input = st.text_area("Enter your sentence:", "Your text here", height=10)
language_level_options = ["Elementary", "High School", "University"]
language_level_input = st.selectbox("Choose the language level:", language_level_options)
# generate button after text input
if st.button('Generate'):
    messages_so_far = [
        {"role": "system", "content": prompt},
        {'role': 'user', 'content': sentence_input},
        {'role': 'user', 'content': language_level_input},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_so_far
    )
    # Show the response from the AI 
    response_dict = response.choices[0].message.content
    sd = json.loads(response_dict)
    original_text = sd["original_text"]
    paraphrased_text = sd["paraphrased_text"]
    vocabulary_list = sd["vocabulary_list"]
    for entry in sd['vocabulary_list']:
        entry['synonyms'] = ', '.join(entry['synonyms'])
    # Create DataFrames
    original_paraphrased_df = pd.DataFrame({"Original Text": [original_text], "Paraphrased Text": [paraphrased_text]})
    vocabulary_df = pd.DataFrame(vocabulary_list)

    # Show DataFrames
    st.markdown('**Original and Paraphrased Sentences:**')
    st.table(original_paraphrased_df)

    st.markdown('**Vocabulary List:**')
    st.table(vocabulary_df)




    