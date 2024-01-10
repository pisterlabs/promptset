from openai import OpenAI
import os
from trulens_eval import TruLlama, FeedbackMode, Feedback, Tru
from trulens_eval.feedback import Groundedness
from trulens_eval import OpenAI as fOpenAI
from llama_index import StorageContext, load_index_from_storage
import streamlit as st
import numpy as np
import google.auth
import pandas as pd
import nest_asyncio
from datetime import datetime
import json

def write_file():
    key_path = st.secrets["JSON_PATH"]
    with open(key_path, "w+") as f:
        creds = {
            "type": st.secrets["type"],
            "project_id": st.secrets["project_id"],
            "private_key_id": st.secrets["private_key_id"],
            "private_key": st.secrets["private_key"],
            "client_email": st.secrets["client_email"],
            "client_id": st.secrets["client_id"],
            "auth_uri": st.secrets["auth_uri"],
            "token_uri": st.secrets["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["client_x509_cert_url"],
            "universe_domain": st.secrets["universe_domain"]
        }
        json.dump(creds, f)


# Authenticate with Google Cloud
write_file()
key_path = st.secrets["JSON_PATH"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
credentials, project_id = google.auth.default()

# Authenticate with OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_AI_API_KEY"]

# Rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="ask_priya_index")

# Load index from the storage context
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()

# Define metrics
provider = fOpenAI()

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

context_selection = TruLlama.select_source_nodes().node.text
f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)

grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

tru = Tru(database_redact_keys = True)
# To clear data base of previous evalutions, uncomment below
# tru.reset_database()

# Init a recorder named by date evaluated
cur_datetime = datetime.now()
cur_datetime_str = cur_datetime.strftime("%Y-%m-%d %H:%M:%S")
tru_recorder = TruLlama(
    query_engine,
    app_id="Engine " + cur_datetime_str,
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)

# Load evaluation questions
eval_questions = []
with open('data/eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)

# Run evaluation engine on each eval question
for question in eval_questions:
    with tru_recorder as recording:
        query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])

# Print performance
print(tru.get_leaderboard(app_ids=[]))

# Open Streamlit dashboard
# nest_asyncio.apply()
# tru.run_dashboard()