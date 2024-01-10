import time
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key = st.secrets.openai.api_key)

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run
