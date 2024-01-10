import streamlit as st
from openai import OpenAI
from colorama import Fore, Style
import time

client = OpenAI(api_key=st.secrets.openai.api_key_general)

def check_run(vThreadId, vRunId):
    while True:
        #refresh run to get latest status
        run = client.beta.threads.runs.retrieve(
            thread_id=vThreadId,
            run_id=vRunId
        )

        if run.status == "completed":
            print(f"{Fore.GREEN} Run is completed. {Style.RESET_ALL}")
            break
        elif run.status == "expired":
            print(f"{Fore.RED} Run is expired. {Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.YELLOW} Run is not yet complete. Waiting...{run.status} {Style.RESET_ALL}")
            time.sleep(3)

