"""
This module handles story initialization. It includes functions for generating a base story
schema, parsing it, saving necessary information to disk, and spawning threads to write suspect accounts.

Functions:
    fix_trailing_commas(json_string: str) -> dict:
        Fixes trailing commas in JSON strings and returns a JSON object.

    save_schema(plot, save_path: str) -> None:
        Saves the plot and suspect schema to a file.

    save_timeline(timeline, save_path: str) -> None:
        Saves the timeline of events to a file.

    save_shared_events(shared_events, save_path: str) -> None:
        Saves shared events to a JSON file.

    save_shared_interactions(plot, save_path: str) -> None:
        Saves shared interactions to a file.

    initialize_story(save_path: str):
        Initializes the story by generating a base story schema, parsing it,
        saving necessary information to disk, and spawning threads to write suspect accounts.
"""

import openai
import os
import re
import json
import threading

# Importing necessary functions and templates from other modules
from LLM_functions.set_up_story import set_up_story_func
from LLM_functions.shared_events import shared_events_func
from LLM_functions.write_timestamp import write_timestamp_func
from .shared_convo_gen import generate_shared_interactions
from .parse_shared_events import parse_shared_interactions
from .suspect_init import write_suspect_account
from .write_timeline import write_timeline
from .parse_story import parse_story
from prompt_templates.story_init import author_template
from prompt_templates.shared_events import shared_events_template


def fix_trailing_commas(json_string: str) -> dict:
    """
    Corrects the trailing commas in a JSON string to ensure it is a valid JSON object.
    Invalid commas are usually found at the end of collections such as lists and dictionaries.

    Args:
        json_string (str): The JSON string to be corrected.

    Returns:
        dict: A valid JSON object after correcting the trailing commas.
    """
    json_string = re.sub(",\s*}", "}", json_string)
    json_string = re.sub(",\s*\]", "]", json_string)
    return json.loads(json_string)


def save_schema(plot, save_path: str) -> None:
    """
    Writes the plot and suspect schema to a file for persistent storage.

    Args:
        plot: The Plot object containing the story schema.
        save_path (str): The directory path where the schema will be saved.
    """
    with open(f"{save_path}/plot_overview.txt", "w") as f:
        f.write(plot.get_summary())


def save_timeline(timeline, save_path: str) -> None:
    """
    Writes the timeline of events to a file for persistent storage.

    Args:
        timeline: The timeline object containing the timestamps.
        save_path (str): The directory path where the timeline will be saved.
    """
    with open(f"{save_path}/timeline.txt", "w") as f:
        f.write(f"TIME OF MURDER: {timeline.time_of_murder}\n\n")
        f.write("TIMELINE:\n")
        for timestamp in timeline.timestamps:
            f.write(f"\nTime: {timestamp.time}")
            for i in range(len(timestamp.suspect_actions)):
                suspect_key = f"suspect_{i + 1}"
                f.write(f"\n\t{timestamp.suspect_actions[suspect_key]}\n")


def save_shared_events(shared_events, save_path: str) -> None:
    """
    Writes the shared events to a JSON file for persistent storage.

    Args:
        shared_events: The shared events object containing the interactions.
        save_path (str): The directory path where the shared events will be saved.
    """
    with open(f"{save_path}/shared_events.txt", "w") as f:
        json.dump(shared_events, f, indent=4)


def save_shared_interactions(plot, save_path: str) -> None:
    """
    Writes the shared interactions to a file for persistent storage.

    Args:
        plot: The plot object containing the shared interactions.
        save_path (str): The directory path where the shared interactions will be saved.
    """
    with open(f"{save_path}/shared_interactions.txt", "w") as f:
        shared_interactions = plot.shared_interactions

        for interaction in shared_interactions:
            if interaction.interaction_content is not None:
                f.write(f"TIME: {interaction.time}\n\n")
                f.write(f"Suspect a: {interaction.suspect_a}\n")
                f.write(f"Suspect b: {interaction.suspect_b}\n\n")
                f.write(f"Interaction content: \n\n{interaction.interaction_content}\n\n")


def initialize_story(save_path: str):
    """
    Orchestrates the story initialization process by generating a base story schema, parsing it,
    saving necessary information to disk, and spawning threads to write suspect accounts.

    Args:
        save_path (str): The directory path where the story elements will be saved.

    Returns:
        The plot object containing the initialized story schema and elements.
    """
    # Generate base story schema
    story_skeleton = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k-0613',
        messages=[{'role': 'system', 'content': author_template}],
        functions=set_up_story_func,
        function_call={'name': 'set_up_story'},
    )

    # Extracting and parsing the story schema
    story_schema = story_skeleton.get('choices', [{}])[0].get('message', {}).get('function_call', {}).get('arguments')
    plot, suspects = parse_story(fix_trailing_commas(story_schema))

    # Saving schema to disk
    save_schema(plot, save_path)

    # Designating timestamps at which shared interactions occur
    story_shared_events = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k-0613',
        messages=[{'role': 'system', 'content': shared_events_template.format(story_schema=story_schema)}],
        functions=shared_events_func,
        function_call={'name': 'shared_events'},
    )

    # Extracting and parsing the shared events
    shared_events = story_shared_events.get('choices', [{}])[0].get('message', {}).get('function_call', {}).get(
        'arguments')
    parse_shared_interactions(fix_trailing_commas(shared_events), plot)

    # Saving shared events and timeline to disk
    save_shared_events(fix_trailing_commas(shared_events), save_path)
    write_timeline(plot, write_timestamp_func)

    # Generating shared interactions
    generate_shared_interactions(plot)
    save_shared_interactions(plot, save_path)

    # Creating a directory for suspect accounts
    accounts_path = os.path.join(save_path, "accounts")
    if not os.path.exists(accounts_path):
        os.mkdir(accounts_path)

    # Spawning threads to write suspect accounts
    threads = []
    for i in range(4):
        print(f"Creating thread for suspect {i + 1}")
        t = threading.Thread(target=write_suspect_account, args=(plot, i, accounts_path))
        threads.append(t)
        t.start()

    # Waiting for all threads to finish
    for t in threads:
        t.join()

    return plot
