from dotenv import load_dotenv
import os
from openai import OpenAI
from datetime import datetime, timedelta, timezone
import time

load_dotenv()

def epoch(epoch_time):
    jst_timezone = timezone(timedelta(hours=9))
    return datetime.fromtimestamp(epoch_time, jst_timezone).strftime("%Y-%m-%d %H:%M:%S")

client = OpenAI()

print("Retrieving all assistants...")
assistants = client.beta.assistants.list(limit=100).data

# Check if there are any assistants to process
if len(assistants) < 1:
    print("No assistants found. Exiting.")
else:
    confirm = input("Are you sure you want to delete all assistants named '孫悟空'? (Yes/No): ")

    if confirm.lower() == 'yes':
        for assistant in assistants:
            print(f"current assistant: {assistant.id}, {assistant.name}, {epoch(assistant.created_at)}")
            if "孫悟空" == assistant.name:
                print("Deleting assistant: ", assistant.id)
                result = client.beta.assistants.delete(assistant_id=assistant.id)
                print(result)
    else:
        print("Deletion cancelled.")
