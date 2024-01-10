import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
"""
This script is used for testing accounts only. 

WARNING: This script will delete all files and assistants in your OpenAI organization.

Run this script to delete all files and assistants in your OpenAI organization.

"""

# TODO:
# - Allow passing a list of assistants and files to delete
api_key = os.getenv("OPENAI_API_KEY")
organization = os.getenv("OPENAI_ORG_KEY")
client = OpenAI(organization=organization, api_key=api_key)


def delete_all_files_assistants():
    try:
        # Collect all file IDs first
        file_ids = [file_obj.id for file_obj in client.files.list()]

        # Now delete each file
        for file_id in file_ids:
            client.files.delete(file_id)
            print(f"Deleted file with ID: {file_id}")
    except Exception as e:
        print(f"An error occurred while deleting files: {e}")

    try:
        # Collect all assistant IDs first
        assistant_ids = [
            assistant.id
            for assistant in client.beta.assistants.list(order="desc", limit="20")
        ]

        # Now delete each assistant
        for assistant_id in assistant_ids:
            client.beta.assistants.delete(assistant_id)
            print(f"Deleted assistant with ID: {assistant_id}")
    except Exception as e:
        print(f"An error occurred while deleting assistants: {e}")


def main():
    proceed = input(
        "type 'DELETE ALL' to delete all files and assistants in your OpenAI organization. "
    )
    if proceed == "DELETE ALL":
        delete_all_files_assistants()
    else:
        print("Exiting without deleting anything.")


main()
