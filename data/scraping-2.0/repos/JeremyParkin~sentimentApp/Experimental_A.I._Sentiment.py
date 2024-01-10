import streamlit as st
import pandas as pd
import mig_functions as mig
import json
import replicate
import io
import time
import openai
from openai import OpenAI
client = OpenAI(api_key=st.secrets["key"])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import string
import re



# Set Streamlit configuration
st.set_page_config(page_title="MIG Sentiment Tool",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png",
                   layout="wide")


# Sidebar configuration
mig.standard_sidebar()


# action_rows = 5
total_time = 0.0  # Initialize the total time taken for all API calls
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
cost_per_sec = 0.001400 # approximate
# SIMILARITY_THRESHOLD = 0.95  # Can be adjusted

st.title("EXPERIMENTAL: Bulk Sentiment")

def normalize_text(text):
    # Convert to string in case the input is not a string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Further normalization steps like stemming could be added here
    return text


def remove_extra_spaces(text):
    # Convert to string in case the input is not a string
    text = str(text)
    # Remove extra spaces from the beginning and end
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text



def calculate_similarity(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Normalized Headline'] + " " + df['Normalized Snippet'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def assign_group_ids(duplicates):
    group_id = 0
    group_ids = {}

    for i, similar_indices in duplicates.items():
        if i not in group_ids:
            group_ids[i] = group_id
            for index in similar_indices:
                group_ids[index] = group_id
            group_id += 1

    return group_ids



def identify_duplicates(similarity_matrix):
    duplicates = {}
    for i in range(similarity_matrix.shape[0]):
        duplicates[i] = []
        for j in range(similarity_matrix.shape[1]):
            if i != j and similarity_matrix[i][j] > similarity_threshold:
                duplicates[i].append(j)
    return duplicates



named_entity = st.text_input("What brand, org, or person should the sentiment be based on?", max_chars=100,
                             key="named_entity")

if len(named_entity) < 1:
    named_entity = "the organization"

analysis_placeholder = "Eg. As an academic institution, news stories covering academic presentations are typically positive or neutral, even if the subject of the presentation is not."
analysis_placeholder2 = f"As a guideline, positive stories might focus on {named_entity}'s good qualities, values, and successes as demonstrated by it and its representatives.  Negative might focus on {named_entity}'s legal or financial difficulties, lawsuits or accusations against it or its representatives, safety issues, racism, abuse, harassment, fraud, or other types of misconduct."

analysis_note = st.text_area("Any special note on sentiment approach?", max_chars=600, key="analysis_notes",
                             help="This will be added as a note in the prompt for each story. Use it as is or feel free to, edit, delete, or replace it as needed.",
                             placeholder=analysis_placeholder, height=80)


c1, c2, c3, c4, c5 = st.columns(5, gap="large")  # Create five columns

with c1:
    sentiment_switch = st.radio("Sentiment Type", ['Simple', '3-way', '5-way'])

with c2:
    random_sample = st.radio('Randomize Sample?', ['No', 'Yes'])

with c3:
    action_rows = st.number_input('Limit rows for testing (max 100)', min_value=1, value=5, max_value=600)

with c4:
    model_choice = st.radio(
        "Choose the AI model:",
        ('LLAMA 2', 'GPT-3.5', 'GPT-4'),
        key='model_choice'
    )

with c5:
    similarity_threshold = st.slider(
        'Similarity Threshold',
        min_value=0.85, max_value=1.0, value=0.95, step=0.01,
        help='Set the threshold for grouping similar articles. Higher values mean more similarity is required.'
    )



# c1, c2, c3, c4 = st.columns(4)
# with c1:
#     sentiment_switch = st.radio("Sentiment Type", ['Simple', '3-way', '5-way'], )
# with c2:
#     random_sample = st.radio('Randomize Sample?', ['No', 'Yes'])
#
# with c3:
#     action_rows = st.number_input('Limit rows for testing (max 100)', min_value=1, value=5, max_value=600)
# with c4:
#     # Add a radio button for model selection
#     model_choice = st.radio(
#         "Choose the AI model:",
#         ('LLAMA 2', 'GPT-3.5', 'GPT-4'),
#         key='model_choice'
#     )

if sentiment_switch == '5-way':
    sentiment_type = "VERY POSITIVE / MODERATELY POSITIVE / NEUTRAL / MODERATELY NEGATIVE / VERY NEGATIVE / UNAVAILABLE"
    sentiment_rubric = f"""
            \nVERY POSITIVE: A positive portrayal of {named_entity}, focusing on its merits, successes, or positive contributions. A positive headline or first sentence is a good clue of a VERY POSITIVE story.
            \nSOMEWHAT POSITIVE: A net positive view of {named_entity}, but with some minor reservations or criticisms, maintaining a supportive stance overall.
            \nNEUTRAL: A passing mention or objective perspective of {named_entity}, balancing both praise and critique without favoring either.
            \nSOMEWHAT NEGATIVE: A mildly negative depiction of {named_entity}, highlighting its shortcomings, but may acknowledge some positive elements.
            \nVERY NEGATIVE: A strongly critical portrayal of {named_entity}, emphasizing significant concerns or failings.
            \nUNAVAILABLE: {named_entity} is not mentioned.
            """

elif sentiment_switch == 'Simple':
    sentiment_type = "POSITIVE / NEUTRAL / NEGATIVE / UNAVAILABLE"
    sentiment_rubric = f"""
            \nPlease indicate the sentiment of the following news story as it relates to {named_entity}. POSITIVE, NEUTRAL, or NEGATIVE (or UNAVAILABLE).
            A passing mention should typically be NEUTRAL.  
            """


else:
    sentiment_type = "POSITIVE / NEUTRAL / NEGATIVE / UNAVAILABLE"
    sentiment_rubric = f"""
            \nPOSITIVE: A positive portrayal of {named_entity}, focusing on its merits and successes.
            \nNEUTRAL: A balanced portrayal of {named_entity} or a passing mention of {named_entity} that is not overtly positive or negative.
            \nNEGATIVE: A critical portrayal of {named_entity}, emphasizing significant concerns or failings.
            \nUNAVAILABLE: {named_entity} is not mentioned.
            """


estimated_cost = action_rows * cost_per_sec * 3
# st.write(f"Estimated cost based on max rows: ${estimated_cost:.2f}")


def prompt_preview():
    prompt_text = f"""
                You are acting as an entity sentiment AI, indicating how a news story portrays {named_entity} based on these options: 
                \n{sentiment_rubric}
                \n{analysis_note}                
                \nHere is the news story...
            """
    return prompt_text


def generate_prompt(row, named_entity):
    prompt_text = f"""
        You are acting as an entity sentiment AI, indicating how a news story portrays {named_entity}.  Respond with only the ONE WORD label based on following options: 
        \n{sentiment_rubric}
        \n{analysis_note} 
        REMEMBER, respond ONLY with the ONE WORD SENTIMENT LABEL, nothing more.
        This is the news story:

        {row['Headline']}. {row['Snippet']}
        """
    return prompt_text





with st.form('User Inputs'):
    upload_file = st.file_uploader("Upload a CSV file:", type=["csv"])

    if upload_file:
        try:
            if upload_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(upload_file)
            else:
                df = pd.read_csv(upload_file)

            if random_sample == 'Yes':
                df = df.sample(frac=1).reset_index(drop=True)

            # Limit the rows after randomizing the sample
            df = df.head(action_rows)

            df = df.rename(columns={'Published Date': 'Date'}, errors='ignore')
            df = df.rename(columns={'Coverage Snippet': 'Snippet'}, errors='ignore')

        except Exception as e:
            st.error("Error reading the file: {}".format(str(e)))
            st.stop()


    submitted = st.form_submit_button("Submit")


st.divider()
st.subheader("Prompt Preview")
st.write(prompt_preview())
st.divider()


if submitted and (upload_file is None or named_entity == "the organization"):
    st.error('Missing required form inputs above.')


elif submitted:
    df['Entity Sentiment'] = ""
    progress = st.progress(0)
    number_of_rows = len(df)
    to_be_done = min(action_rows, number_of_rows)

    df['Headline'] = df['Headline'].apply(remove_extra_spaces)
    df['Snippet'] = df['Snippet'].apply(remove_extra_spaces)
    df['Normalized Headline'] = df['Headline'].apply(normalize_text)
    df['Normalized Snippet'] = df['Snippet'].apply(normalize_text)

    # st.dataframe(df)
    st.write(f"Total Stories: {len(df)}")

    similarity_matrix = calculate_similarity(df)
    # st.write("Sample Similarity Matrix:", similarity_matrix[:5, :5])  # Display a 5x5 sample of the matrix

    duplicate_groups = identify_duplicates(similarity_matrix)


    # Assign group IDs
    group_ids = assign_group_ids(duplicate_groups)
    df['Group ID'] = df.index.map(group_ids)

    # Group by 'Group ID'
    unique_stories = df.groupby('Group ID').first().reset_index()





    # st.dataframe(unique_stories)
    st.write(f"Unique Stories: {len(unique_stories)}")

    # Dictionary to store sentiment for each group
    group_sentiments = {}

    for i, row in unique_stories.iterrows():
        if pd.isna(row['Snippet']) or len(row['Snippet']) < 350:
            st.warning(f"Snippet is too short for story {i + 1}")
            progress.progress((i + 1) / to_be_done)
            continue
        # if len(row['Snippet']) > 9250:
        #     st.warning(f"Snippet is too long for story {i + 1}")
        #     progress.progress((i + 1) / to_be_done)
        #     continue

        story_prompt = generate_prompt(row, named_entity)

        # Conditional to choose the API call based on model choice
        if model_choice == 'LLAMA 2':
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
            model_call = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781"

            try:
                start_time = time.time()  # Record the start time

                retries = 0
                while retries < MAX_RETRIES:
                    output = replicate.run(
                        model_call,
                        input={
                            "prompt": story_prompt,
                            # "system_prompt": "You are a highly knowledgeable media analysis AI."
                        }
                    )

                    # Assuming the output is a list of strings and we take the first item.
                    sentiment = ''.join(output).strip()
                    df.at[i, 'Entity Sentiment'] = sentiment

                    break  # If successful, break out of the retry loop

            except (TimeoutError, Exception) as e:
                retries += 1
                if retries == MAX_RETRIES:
                    st.error(f"Error processing message {i + 1} after {MAX_RETRIES} retries: {str(e)}")

                else:
                    time.sleep(RETRY_DELAY)  # Wait before retrying





        # elif model_choice == 'GPT-3.5':
        elif model_choice == 'GPT-3.5' or model_choice == 'GPT-4':
            if model_choice == 'GPT-3.5':
                model_id = "gpt-3.5-turbo-1106"
            elif model_choice == 'GPT-4':
                model_id = "gpt-4-1106-preview"

            openai.api_key = st.secrets["key"]
            try:
                response = client.chat.completions.create(
                    model = model_id,
                    messages=[
                        {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                        {"role": "user", "content": story_prompt}
                    ]
                )
                # sentiment = response['choices'][0]['message']['content'].strip()
                # st.write(response)
                sentiment = response.choices[0].message.content
                df.at[i, 'Entity Sentiment'] = sentiment

                time.sleep(2)  # Add a 1-second delay here

            except openai.RateLimitError:
                st.warning("Rate limit exceeded. Waiting for 20 seconds before retrying.")
                time.sleep(20)
            except openai.APIError as e:
                st.error(f"Error while processing the request: {e}")
                st.error("Please check the request ID and contact OpenAI support if the error persists.")
                break  # Break the loop if there's an API error

            except Exception as e:  # Catch-all for any other exceptions
                st.error(f"An unexpected error occurred: {e}")

        # Update the sentiment in the original DataFrame
        # For all stories that belong to the same group as the current unique story
        group_id = row['Group ID']
        group_sentiments[group_id] = sentiment
        # df.loc[df['Group ID'] == group_id, 'Entity Sentiment'] = sentiment

        # Update progress bar
        progress.progress((i + 1) / len(unique_stories))

    # After processing all unique stories, assign sentiments to all stories
    for group_id, sentiment in group_sentiments.items():
        df.loc[df['Group ID'] == group_id, 'Entity Sentiment'] = sentiment


    # st.dataframe(df)

    required_cols = ['Headline', 'Snippet', 'Sentiment', 'Entity Sentiment', 'Group ID']
    df_display = df.filter(required_cols)
    st.dataframe(df_display, hide_index=True)

    # Create a download link for the DataFrame as an Excel file
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()  # Use writer.close() instead of writer.save()
    output.seek(0)
    st.download_button(
        label="Download sentiment file",
        data=output,
        file_name="entity_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type='primary'
    )