from robocorp.tasks import task
from robocorp import vault, storage, excel

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate

import json

@task
def compare_addresses():
    """Read address pairs from an excel file, and compare their similiarity using OpenAI.
    The prompt template and OpenAI API credentials are stored in Robocorp Control Room.
    Inspired by the blog post by Benjamin Stein: https://www.haihai.ai/programming-with-llm/"""

    # Get addresses from an Excel (example comes with the repo)
    addresses = excel.open_workbook("addresses.xlsx").worksheet("Sheet1").as_list(header=True)

    # Set up LLM using credentials from Robocorp Vault - edit to match your own entries
    openai_credentials = vault.get_secret("OpenAI")
    llm = ChatOpenAI(openai_api_key=openai_credentials["api-key"])

    # Create the prompt template using Robocorp Asset Storage, easy to edit the prompt in the cloud
    # without deploying code changes. Edit the name of the Asset to match your own.
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=("You are a helpful assistant that compares addresses for the user.")),
            HumanMessagePromptTemplate.from_template(storage.get_text("example_prompt_template")),
        ]
    )

    for row in addresses:
        print(f"\nComparing addresses: {row['First address']} to {row['Second address']}.")

        # Run LLM chat completion by feeding in the addresses to the template
        response = llm(template.format_messages(address_one=row["First address"], address_two=row["Second address"]))
        response_json = json.loads(response.content)
        print(f"RESULT: {response_json['result']}, because {response_json['reason']}")
