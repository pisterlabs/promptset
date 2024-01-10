"""
This module is responsible for writing a narrative account for each suspect involved in a story plot. It sets up the
necessary objects and templates required to generate a narrative, constructs narrative headers, and writes suspect
accounts to disk.

Functions:
    get_narrative_header(plot, suspect_id: int) -> str:
        Constructs and returns the narrative header for a suspect's account.

    write_suspect_account(plot, suspect_id: int, accounts_path: str) -> None:
        Generates, writes a suspect's account to a file.
"""

import openai
import os

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from prompt_templates.generate_perspective import generate_perspective_template
from .assemble_suspect_context import assemble_suspect_context

openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a memory object
memory = ConversationBufferMemory()

# Create an LLM object for chain
llm = ChatOpenAI(
    temperature=0.45,
    model="gpt-3.5-turbo-16k-0613",
)

prompt = PromptTemplate.from_template(generate_perspective_template)

# Create a chain object
chain = LLMChain(
    llm=llm,
    memory=memory,
    verbose=False,
    prompt=prompt,
)

info_template = """Here is some background information about the case:
{character_description}

Here is the part of the event you are currently living out again in your dream:
{event}"""


def get_narrative_header(plot, suspect_id):
    """
    Constructs and returns the narrative header for a suspect's account.
    """

    print(f"ID: {suspect_id + 1}")

    suspect = plot.suspects[suspect_id]

    # Create an empty string
    output_str = ""

    # Add each line of output to the string, using the 'suspect' variable
    output_str += f"Suspect name: {suspect.name}\n"
    output_str += f"Suspect bio: {suspect.bio}\n"
    output_str += f"Suspect tags: {suspect.tags}\n"
    output_str += f"Suspect victim connection: {suspect.victim_connection}\n"
    output_str += f"Guilty?: {suspect.guilty}\n"
    output_str += f"Suspect ID: {suspect.id}\n"

    # Return the string
    return output_str


def write_suspect_account(
        plot,
        suspect_id,
        accounts_path):
    """
    Generates, writes a suspect's account to a file.
    """
    # Create 'accounts' folder if it does not exist
    suspect_account_path = f"{accounts_path}/suspect_{suspect_id + 1}"

    # Write a suspect account
    with open(suspect_account_path, "w") as f:
        suspect_context = assemble_suspect_context(plot, suspect_id)

        header = get_narrative_header(plot, suspect_id)
        f.write(header)

        # Go through each timestamp and write an account of the suspect's action
        f.write("\nTIMELINE:\n\n")
        for i, timestamp in enumerate(plot.timeline):
            action = timestamp.suspect_actions[suspect_id]
            time_and_action = f"At {timestamp.time}, {action}"

            suspect = plot.suspects[suspect_id]
            suspect.memory_path = suspect_account_path

            # If there is a shared interaction, use it to inform the suspect's account
            shared_interaction = plot.shared_interactions[i]
            if shared_interaction.interaction_content is not None:
                if suspect.name == shared_interaction.suspect_a or suspect.name == shared_interaction.suspect_b:
                    time_and_action += f"This is how the interaction went:"
                    time_and_action += f"\n\n{shared_interaction.interaction_content}"

            # Format prompt template; prompt the model, write output to the file
            info = info_template.format(character_description=suspect_context, event=time_and_action)

            entry = chain(info)
            f.write(f"\n\nTIME: {timestamp.time}\n")
            f.write(entry['text'])
