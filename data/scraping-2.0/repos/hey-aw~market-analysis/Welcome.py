from curses import meta
import os
import time
import json
import streamlit as st
import pandas as pd
import openai
from langchain.document_loaders import Docx2txtLoader

st.title("Hello")

"""
# Primary Goal
To understand what and why life science educators in high school buy from Carolina and/or Flinn.

# Questions
- What is Carolina most known, loved and trusted for?
- What are the attributes of reliability?
- What are the attributes of trust for life science educators?
- How do Carolina, Flinn & Amazon rank on the attributes?

# Client Uses
- Foundation work for customer journey
- Market positioning
- Brand development based on key attributes
- Market mapping
- Competitive positioning
"""
# 1. Load transcripts
# Load transcripts from /data directory

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
TRANSCRIPT_PATH = "data/transcripts"
ASSISTANT_ID = "asst_XfSqQlAPl7opg5fsMCjE9lfL"
transcript_files = os.listdir(TRANSCRIPT_PATH)
transcripts_df = pd.DataFrame()

# Function to upload a file for assistant
def upload_file(file_path):
    # Upload a file with an "assistants" purpose
    file = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    return file.id

# 2. Code the transcripts with Assistants API
client = openai.Client()

status_indicator = st.status("Loading transcripts...")


@st.cache_data
def add_transcript_with_annotations(data):
    transcript = json.loads(data)
    st.dataframe(transcript)
    # add transcript to dataframe
    transcripts_df.append(transcript, ignore_index=True)
    return data


def update_status_indicator(run):
    status_indicator.update(label=run.status)
    status_indicator.empty()
    status_indicator.write(run)


@st.cache_data
def request_annotations_from_assistants_api(file_path):
    transcript_file_id = upload_file(file_path)
    
    def poll_run(thread_id, run_id):
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        while run.status != "completed" and run.status != "requires_action":
            # wait 1 second before polling again
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            update_status_indicator(run)
        update_status_indicator(run)
        return run

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Please process this transcript.",
        file_ids=[transcript_file_id],
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=ASSISTANT_ID
    )
    update_status_indicator(run)
    # poll the run until it requires action or completes
    run = poll_run(thread.id, run.id)

    # if the run requires action, get the action
    if run.status == "requires_action" or "completed":
        # get the tool calls from the run
        if run.required_action and run.required_action.submit_tool_outputs:
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            # look for the add_transcript_with_annotations tool call
            for tool_call in tool_calls:
                function = tool_call.function
                if function and function.name == "add_transcript_with_annotations":
                    # get the transcript with annotations
                    args = function.arguments
                    response = add_transcript_with_annotations(args)
                    # submit a successful response to tool call
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=[
                            {
                                "tool_call_id": tool_call.id,
                                "output": str(response),
                            }
                        ],
                    )
                    return args

        
metadata_df = pd.read_csv('./data/survey_data.csv')

def normalize_last_name(last_name):
    # normalize last name
    last_name = last_name.lower().strip()
    # remove special characters
    last_name = last_name.replace("-", "'")
    return last_name
# test that last name field is in metadata
last_name = "Parker"
metadata_df
# Match with normalized last name
metadata_df = metadata_df[normalize_last_name(metadata_df["RecipientLastName"]) == normalize_last_name(last_name)]

# if there is no match, as for the survey code
if metadata_df.empty:
    survey_code = st.text_input("Please enter your survey code")
    metadata_df = metadata_df[metadata_df["SurveyCode"] == survey_code]

# if there is a match, combine the transcript and survey data

# for file in transcript_files[:1]:
#     file_path = os.path.join(TRANSCRIPT_PATH, file)
#     transcript = request_annotations_from_assistants_api(file_path)


# 3. Analyze the transcripts
## Word cloud
## Sentiment analysis
## Topic modeling
