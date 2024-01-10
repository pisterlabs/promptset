import pandas as pd
import streamlit as st
import warnings
import mig_functions as mig
import openai
import time
from datetime import datetime
import re
from openai import OpenAI


type_dict = {
    "RADIO": "broadcast transcript",
    "TV": "broadcast transcript",
    "PODCAST": "broadcast transcript",
    "ONLINE": "online article",
    "PRINT": "print article",
}

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(layout="wide", page_title="MIG Data Processing App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png")

# Standard sidebar
mig.standard_sidebar()

# Page title
st.title("Top Stories Summaries")

if not st.session_state.upload_step:
    st.error('Please upload a CSV before trying this step.')
elif not st.session_state.standard_step:
    st.error('Please run the Standard Cleaning before trying this step.')
elif len(st.session_state['added_df']) == 0:
    st.error('Please select your TOP STORIES before trying this step.')

else:
    client = OpenAI(api_key=st.secrets["key"])

    # Load the DataFrame with top stories
    df = st.session_state.added_df


    # Form for user input
    with st.form('User Inputs'):
        # User input for analysis mode and named entity
        mode = st.radio("Analysis Mode", ('Summary', 'Sentiment'), help='Produce a short summary or sentiment opinion relative to the client brand for the selected top stories.')
        named_entity = st.text_input("Enter the named entity:", max_chars=100, key="named_entity",
                                     help="The brand, organization, or person this analysis should focus on",
                                     value=st.session_state.client_name)

        submitted = st.form_submit_button("Submit")

    # Check if the form is submitted and named entity is not empty
    if submitted and named_entity:
        # Set OpenAI API key
        openai.api_key = st.secrets["key"]

        # Define function to generate summary prompt
        def generate_summary_prompt(row, named_entity):

            if row['Example Type'] == "RADIO" or "TV":
                summary_prompt = f"""
                Provide an executive content analysis of {named_entity} in the following broadcast transcript. Note that
                broadcast transcripts often contain clips of unrelated advertisements and other segments that should be ignored.
                The summary should be concise, approximately 20-25 words, and should not include any 
                labels or introductory text (not even the word 'Summary'). 
                \n\nHEADLINE:\n{row['Headline']}\n BODY: \n{row['Example Snippet']}"
                """


            else:
                summary_prompt = f"""
                   Provide an executive content analysis of {named_entity} in the following {type_dict[row['Example Type']]}. 
                   The summary should be concise, approximately 20-25 words, and should not include any 
                   labels or introductory text (not even the word 'Summary'). 
                   \n\nHEADLINE:\n{row['Headline']}\n BODY: \n{row['Example Snippet']}"
                   """


            return summary_prompt


        # Define function to generate sentiment prompt
        def generate_sentiment_prompt(row, named_entity):
            return f"Please indicate the sentiment of the following news story as it relates to {named_entity}. Start with one word: Positive, Neutral, or Negative - followed by a colon then a one sentence rationale as to why that sentiment was chosen.\n\nThis is the news story:\n{row['Headline']}. {row['Example Snippet']}"


        # Initialize progress bar
        progress_bar = st.progress(0)
        total_items = len(df)  # Get the total number of items to process
        processed_items = 0  # Initialize the number of processed items

        # Loop through each row in the DataFrame to generate summaries or sentiments
        for i, row in df.iterrows():
            try:
                # Check if the snippet is long enough to generate a response
                if len(row['Example Snippet']) < 150:
                    st.warning(f"Snippet is too short for message {i + 1}.")
                    if mode == "Summary":
                        df.at[i, 'Entity Summary'] = "Snippet too short to generate summary"
                    else:  # Sentiment mode
                        df.at[i, 'Entity Sentiment'] = "Snippet too short to generate sentiment opinion"

                    continue

                # Generate prompt based on mode
                if mode == "Summary":
                    prompt = generate_summary_prompt(row, named_entity)
                    df.at[i, 'Entity Summary'] = ""
                else:  # Sentiment mode
                    prompt = generate_sentiment_prompt(row, named_entity)
                    df.at[i, 'Entity Sentiment'] = ""

                # Call the OpenAI API using the chat interface
                # response = openai.ChatCompletion.create(
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                        {"role": "user", "content": prompt}
                    ]
                )

                # Update progress bar after each item is processed
                processed_items += 1
                progress = int((processed_items / total_items) * 100)
                progress_bar.progress(progress)


                # Update the DataFrame with the response
                if mode == "Summary":
                    summary = response.choices[0].message.content.strip()
                    df.at[i, 'Entity Summary'] = summary
                else:
                    sentiment = response.choices[0].message.content.strip()
                    df.at[i, 'Entity Sentiment'] = sentiment




            except openai.RateLimitError:
                st.warning("Rate limit exceeded. Waiting for 20 seconds before retrying.")
                time.sleep(20)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


        # Complete the progress bar when done
        progress_bar.progress(100)



    else:
        if submitted and not named_entity:
            st.error('Named entity is required to proceed.')


    markdown_content = ""


    def escape_markdown(text):
        # List of Markdown special characters to escape
        # markdown_special_chars = r"\`*_{}[]()#+-.!$"
        # escaped_text = re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", text)
        # return escaped_text
        # List of Markdown special characters to escape
        markdown_special_chars = r"\`*_{}[]()#+-.!$"
        # Correctly form the regular expression pattern
        pattern = r"([" + re.escape(markdown_special_chars) + r"])"
        escaped_text = re.sub(pattern, r"\\\1", text)
        return escaped_text

    st.write(" ")

    for story in df.index:
        head = escape_markdown(df.Headline[story])
        outlet = escape_markdown(df["Example Outlet"][story])
        link = escape_markdown(df["Example URL"][story])
        date = df["Date"][story].strftime("%B %d, %Y")

        # Build the markdown string
        markdown_content += f"__[{head}]({link})__  \n"
        markdown_content += f"_{outlet}_ – {date}  \n"

        if "Entity Summary" in df.columns:
            entity_summary = df["Entity Summary"][story]
            markdown_content += f"{entity_summary}  \n\n"

        if "Entity Sentiment" in df.columns:
            entity_sentiment = df["Entity Sentiment"][story]
            markdown_content += "<br>"
            markdown_content += f"_{entity_sentiment}_  \n\n"

        markdown_content += "<br>"

    # Display the entire content as Markdown
    st.markdown(markdown_content, unsafe_allow_html=True)