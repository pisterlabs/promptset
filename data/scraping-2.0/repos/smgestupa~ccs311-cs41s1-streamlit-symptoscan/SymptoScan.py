import time
import random
import openai
import spacy
import regex as re
import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = st.secrets['openai_secret_key']

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

diseases_df = pd.read_csv("https://raw.githubusercontent.com/smgestupa/ccs311-cs41s1-streamlit-symptoscan/main/datasets/diseases.csv")
symptoms_df = pd.read_csv("https://raw.githubusercontent.com/smgestupa/ccs311-cs41s1-streamlit-symptoscan/main/datasets/symptoms.csv")

st.set_page_config(
    page_title="SymptoScan",
    page_icon="🩺"
)

random_quotes = [
    "“Time and health are two precious assets that we don't recognize and appreciate until they have been depleted.” - Denis Waitley",
    "“A fit body, a calm mind, a house full of love. These things cannot be bought - they must be earned.” - Naval Ravikant",
    "“A good laugh and a long sleep are the best cures in the doctor's book.” - Irish proverb",
    "“Let food be thy medicine and medicine be thy food.” - Hippocrates",
    "“A sad soul can be just as lethal as a germ.” - John Steinbeck",
    "“Good health is not something we can buy. However, it can be an extremely valuable savings account.” - Anne Wilson Schaef",
    "“Health is not valued until sickness comes.” - Thomas Fuller",
    "“Your body hears everything your mind says.” - Naomi Judd",
    "“The way you think, the way you behave, the way you eat, can influence your life by 30 to 50 years.” - Deepak Chopra",
    "“If you're happy, if you're feeling good, then nothing else matters.” - Robin Wright",
    "“The first wealth is health.” - Ralph Waldo Emerson"
]

st.sidebar.success(random.choice(random_quotes))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_state" not in st.session_state:
    st.session_state.current_state = "NOT_ASKING"

if "user_threads" not in st.session_state:
    st.session_state.user_threads = []

if "bot_threads" not in st.session_state:
    st.session_state.bot_threads = []

if "possible_diseases" not in st.session_state:
    st.session_state.possible_diseases = []

if "current_symptom" not in st.session_state:
    st.session_state.current_symptom = []

if "experiencing_symptoms" not in st.session_state:
    st.session_state.experiencing_symptoms = []

if "disable_chat_input" not in st.session_state:
    st.session_state.disable_chat_input = False

if "last_symptom" not in st.session_state:
    st.session_state.last_symptom = None

def write_bot_message(response):
    st.session_state.bot_threads.append(response)

    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ""

        for character in response:
            full_response += character
            time.sleep(0.025)

            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        time.sleep(0.01)
        
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})

def get_most_similar_diseases(prompt):
    diseases_text = diseases_df.to_csv(index=False, sep=',')
    
    completion = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': f'I want you to act like a system that produces ONLY THE RESULT, NO PLACEHOLDERS, NOTHING MORE NOTHING LESS. I have this CSV:\n{diseases_text}\nWhat are the closest top 3 diseases based on this prompt, and get as CSV with intact headers and get the index of the results from the given CSV and add it onto a column before "Disease" and the name of the column is "row_index": {prompt}\nIf no similar data is found, simply return FALSE instead. And double check the CSV format, please fix it before sending.'}
        ]
    )

    result = completion.choices[0].message.content
    result_df = pd.read_csv(StringIO(result))

    row_indeces = result_df['row_index']
    similar_diseases = result_df.drop(columns=['row_index'])
    
    responses = []
    for (_, row_index), (_, similar_disease) in zip(row_indeces.items(), similar_diseases.iterrows()):
        responses.append([row_index, similar_disease])

    return responses

def get_disease_response(disease_name):
    completion = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': f'I want you to act like a system that produces ONLY THE RESULT, NO PLACEHOLDERS, NOTHING MORE NOTHING LESS. What is a better response if a patient has {disease_name}? Please expand the response in a way where the patient can be relieved and follow.'}
        ]
    )

    result = completion.choices[0].message.content

    return result

def get_most_similar_response(df, column, query, top_k=1):
    # Remove special characters
    special_chars_pattern = r'[^a-zA-z0-9\s(\)\[\]\{\}]'
    query = re.sub(special_chars_pattern, '', query.strip())

    # Remove stop words and specific POS tags
    doc = nlp(query)

    remove_pos = ["PRON", "PROPN", "AUX", "CCONJ", "NUM"]
    filtered_query = ' '.join([token.text for token in doc if not token.is_stop or token.pos_ not in remove_pos])
      
    # Prepare data
    vectorizer = TfidfVectorizer(use_idf=True, max_df=0.5,min_df=1, ngram_range=(1,3))
    all_data = list(df[column]) + [filtered_query]

    # Vectorize with TF-IDF
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Pick the Top k response
    sorted_indeces = similarity_scores.argsort()[0][::-1][:top_k]

    # Get the similarity score of the chosen response
    similarity_score = similarity_scores[0][similarity_scores.argsort()[0][::-1][:top_k]][0] * 100

    # Fetch the corresponding response
    most_similar_responses = df.iloc[sorted_indeces][column].values

    responses = []
    for response in most_similar_responses:
      response_row = df.loc[df[column] == response]
      responses.append([response_row.index.item(), response_row.to_numpy()[0]])

    return responses, similarity_score

def summarize_chat_threads():
    user_threads = "\n----SEPERATE MESSAGE----\n".join(st.session_state.user_threads)
    bot_threads = "\n----SEPERATE MESSAGE----\n".join(st.session_state.bot_threads)

    completion = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': f'I want you to act like a system that produces ONLY THE RESULT, NO PLACEHOLDERS, NOTHING MORE NOTHING LESS. Analuze and summarize the chat threads between the user and chatbot to gain insights about the user.\n\nThe user threads:\n{user_threads}\n\nThe bot threads:\n{bot_threads}\n\nPlease expand the response.'}
        ]
    )

    result = completion.choices[0].message.content

    return result

def disable_chat_input():
    st.session_state.disable_chat_input = True


"""# 🩺 SymptoScan"""

"""SymptoScan, derived from "symptom" and "scan," is designed to analyze user symptoms and suggest potential diseases/illnesses. Users can input various symptoms, allowing the chatbot to identify or provide insights into potential sicknesses."""

"""**IMPORTANT ADVISORY**: If doubts persist, consulting a licensed doctor is recommended, as they possess the expertise needed for accurate diagnoses, unlike the chatbot relying on internet-based knowledge."""

st.divider()

if len(st.session_state.messages) == 0:
    st.session_state.messages.append({
        'role': 'assistant', 
        'content': "Greetings! I am SymptoScan, your dedicated healthcare companion, here to guide you on your wellness journey. Think of me not merely as a chatbot, but as your very own Baymax-inspired health assistant.\n\n**🤗 Caring Conversations**: Describe your symptoms, and I'll provide information and support.\n\n**🚑 Healthcare Companion**: Much like Baymax's round-the-clock availability, I'm here for you 24/7 and I'm just a message away.\n\n**💊 Educational and Reassuring Insights**: I'm not just here for information; I'm here to educate and reassure. Gain insights into your health conditions and receive guidance.\n\n**🔒 Privacy and Security**: Your health information is as precious as for healthcare capabilities. Rest assured, your data is safe and secure for I don't store any information about you.\n\n**NOTE**: Data, such as messages, do not save and will disappear once the page/browser closes."
    })

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


current_state = st.session_state.current_state
user_threads = st.session_state.user_threads
bot_threads = st.session_state.bot_threads
possible_diseases = st.session_state.possible_diseases
current_symptom = st.session_state.current_symptom
experiencing_symptoms = st.session_state.experiencing_symptoms
last_symptom = st.session_state.last_symptom

if len(user_threads) > 15:
    st.session_state.user_threads.pop()

if len(bot_threads) > 15:
    st.session_state.bot_threads.pop()


if prompt := st.chat_input('Ask away!', disabled=st.session_state.disable_chat_input, on_submit=disable_chat_input):
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    st.session_state.user_threads.append(prompt)


if current_state == "NOT_ASKING" and prompt is not None:
    if prompt in ['help', 'Help']:
        write_bot_message('Good day! You can start or continue this chat by telling us what symptoms you are currently experiencing.\n\nIt would help us if you specify what symptoms: e.g. "I am experiencing symptoms such as runny nose, coughing, sore throat."')

    elif prompt in ['summarize', 'Summarize']:
        if len(bot_threads) >= 5:
            write_bot_message(summarize_chat_threads())
        else:
            write_bot_message("This thread is still short for the bot to summarize. Please converse more with SymptoScan.")
    
    else:
        disease, disease_similarity_score = get_most_similar_response(diseases_df, 'Disease', prompt)

        if disease_similarity_score >= 50:
            row_index, row = disease[0]
            
            write_bot_message(f'Based on the symptoms you are experiencing, you may be experiencing {row[0]}. Symptoms of {row[0]} include: {row[2]}. Is the diagnosis correct?\n\n(Type **Yes** if correct, **No** if wrong, **Stop** if you want to be re-diagnosed, **Summarize** if you want to get the summary of this chat.)')
            st.session_state.current_state = "IS_ASKING"
        else:
            responses = get_most_similar_diseases(prompt)

            if responses == "FALSE":
                write_bot_message(f'We have failed to scan your symptoms, please try again and we recommend listing out what symptoms you are experiencing.\n\n(e.g. I am experiencing symptoms such as runny nose, coughing, sore throat.)')
            
            else:
                row_index, row = responses[0]

                write_bot_message(f'Based on the symptoms you are experiencing, you may be experiencing {row[0]}. Symptoms of {row[0]} include: {row[2]}. Is the diagnosis correct?\n\n(Type **Yes** if correct, **No** if wrong, **Stop** if you want to be re-diagnosed, **Summarize** if you want to get the summary of this chat.)')
                st.session_state.current_state = "IS_ASKING"
                st.session_state.possible_diseases = responses

    st.session_state.disable_chat_input = False
    st.rerun()

elif current_state == "IS_ASKING" and prompt is not None and prompt in ["summarize", "Summarize"] and len(bot_threads) >= 5:
    st.session_state.current_state = "NOT_ASKING"
    st.session_state.possible_diseases = []
    st.session_state.current_symptom = []
    st.session_state.experiencing_symptoms = []

    write_bot_message(summarize_chat_threads())
    
    st.session_state.disable_chat_input = False
    st.rerun()

elif current_state == "IS_ASKING" and prompt is not None and prompt in ["summarize", "Summarize"] and len(bot_threads) < 5:
    st.session_state.current_state = "NOT_ASKING"
    st.session_state.possible_diseases = []
    st.session_state.current_symptom = []
    st.session_state.experiencing_symptoms = []

    write_bot_message("This thread is still short for the bot to summarize. Please converse more with SymptoScan.")
    
    st.session_state.disable_chat_input = False
    st.rerun()

elif current_state == "IS_ASKING" and prompt is not None and prompt in ["stop", "Stop"]: 
    st.session_state.current_state = "NOT_ASKING"
    st.session_state.possible_diseases = []
    st.session_state.current_symptom = []
    st.session_state.experiencing_symptoms = []

    write_bot_message('You can continue this chat by telling us what symptoms you are currently experiencing.\n\nIt would help us if you specify what symptoms: e.g. "I am experiencing symptoms such as runny nose, coughing, sore throat."')
    
    st.session_state.disable_chat_input = False
    st.rerun()

elif current_state == "IS_ASKING" and prompt is not None and prompt in ["yes", "Yes"]:
    row_index, row = possible_diseases[0]

    st.session_state.current_state = "NOT_ASKING"
    st.session_state.possible_diseases = []
    st.session_state.current_symptom = []
    st.session_state.experiencing_symptoms = []

    recommendation = get_disease_response(row[0])

    write_bot_message(f'Glad we got it correct! You are experiencing {row[0]}. {row[3]}.\n\nThe symptoms include, which more than one of these you are currently experiencing: {row[2]}. Our recommendation: {recommendation}')
    
    st.session_state.disable_chat_input = False
    st.rerun()

elif current_state == "IS_ASKING" and prompt is not None and prompt not in ["yes", "Yes"]:
    st.session_state.current_state = "ASKING_SYMPTOM"

    if len(st.session_state.possible_diseases) - 1 <= 0:
        st.session_state.current_state = "SCAN_FAILED"
        st.rerun()

    else:
        st.session_state.possible_diseases.pop(0)
        row_index, row = st.session_state.possible_diseases[0]
        st.session_state.current_symptom = [row_index, row[2].split(", ")]

        st.session_state.disable_chat_input = False
        st.rerun()

elif current_state == "SCAN_FAILED":
    st.session_state.current_state = "NOT_ASKING"
    st.session_state.possible_diseases = []
    st.session_state.current_symptom = []
    st.session_state.experiencing_symptoms = []
    st.session_state.disable_chat_input = False

    write_bot_message(f'We have failed to scan your symptoms, please try again and we recommend listing out what symptoms you are experiencing: e.g. "I am experiencing symptoms such as runny nose, coughing, sore throat."')

    st.rerun()
    
elif current_state == "WAITING_SYMPTOM_CALCULATION":
    likelihood = (len(experiencing_symptoms) / len(current_symptom)) * 100
    
    if likelihood >= 60.0:
        row_index, row = st.session_state.possible_diseases[0]

        st.session_state.current_state = "NOT_ASKING"
        st.session_state.possible_diseases = []
        st.session_state.current_symptom = []
        st.session_state.experiencing_symptoms = []

        recommendation = get_disease_response(row[0])

        write_bot_message(f'You might be experiencing {row[0]}. {row[3]}.\n\nThe symptoms include, which more than one of these you are currently experiencing: {row[2]}. Our recommendation: {recommendation}.\n\n(If you are not confident in our answer, please try again and we recommend listing out what you are experiencing.)')
    
    else:
        st.session_state.current_state = "ASKING_SYMPTOM"
        st.session_state.experiencing_symptoms = []

        row_index, row = st.session_state.possible_diseases[0]
        st.session_state.current_symptom = [row_index, row[2].split(", ")]

    st.session_state.disable_chat_input = False
    st.rerun()

elif current_state == "ASKING_SYMPTOM" and len(possible_diseases) > 0:
    if last_symptom == None:
        st.session_state.last_symptom = current_symptom
        st.session_state.disable_chat_input = True
        st.rerun()

    if len(current_symptom[1]) == 0 and len(st.session_state.possible_diseases) - 1 == 0:
        st.session_state.current_state = "SCAN_FAILED"
        st.rerun()

    elif len(current_symptom[1]) == 0:
        row_index, row = st.session_state.possible_diseases.pop(0)
        st.session_state.current_symptom = [row_index, row[2].split(", ")]

        st.session_state.current_state = "WAITING_SYMPTOM_CALCULATION"
        st.rerun()

    row_index, symptoms = current_symptom
    response, _ = get_most_similar_response(symptoms_df, 'Symptom', symptoms[0])

    write_bot_message(f'Are you experiencing: {symptoms[0].capitalize()}? {response[0][1][1]}.')

    st.session_state.current_state = "WAITING_SYMPTOM_ANSWER"

    st.session_state.disable_chat_input = False
    st.rerun()
    
elif current_state == "WAITING_SYMPTOM_ANSWER" and prompt in ["stop", "Stop"]:
    st.session_state.current_state = "NOT_ASKING"
    st.session_state.possible_diseases = []
    st.session_state.current_symptom = []
    st.session_state.experiencing_symptoms = []

    write_bot_message('You can continue this chat by telling us what symptoms you are currently experiencing.\n\nIt would help us if you specify what symptoms: e.g. "I am experiencing symptoms such as runny nose, coughing, sore throat."')
    
    st.session_state.disable_chat_input = False
    st.rerun()
    
elif current_state == "WAITING_SYMPTOM_ANSWER" and prompt is not None:
    st.session_state.current_state = "ASKING_SYMPTOM"

    if prompt in ["yes", "Yes"]:
        experiencing_symptoms.append(st.session_state.current_symptom[1].pop(0))
    else:
        st.session_state.current_symptom[1].pop(0)

    st.session_state.last_symptom = None
    st.rerun()