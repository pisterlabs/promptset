#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def initialize_chat_model(api_key):
    chat_model = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-16k",
        max_tokens=10000,
        openai_api_key=api_key,
        request_timeout=1500
    )
    return chat_model


def create_system_message_prompt(template):
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    return system_message_prompt


def create_human_message_prompt(template):
    human_message_prompt = HumanMessagePromptTemplate.from_template(template)
    return human_message_prompt


def create_chat_prompt(system_message_prompt, human_message_prompt):
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt


def extract_dialogues_from_chapter(chapter_file, chat_chain):
    chunk_size = 5000
    overlap = 500
    with open(chapter_file, 'r') as file:
        chapter_text = file.read()

    chunks = []
    start = 0
    end = chunk_size
    while start < len(chapter_text):
        chunks.append(chapter_text[start:end])
        start = end - overlap
        end = start + chunk_size

    dialogues = []
    for chunk in chunks:
        response = chat_chain.run(chnk=chunk)
        dialogues.append(response)

    return dialogues


def process_dialogues(dialogues):
    dialogue_collection_raw = []

    for diag in dialogues:
        tmp = diag.split("\n")
        for strngs in tmp:
            dialogue_collection_raw.append(strngs)

    return dialogue_collection_raw


def write_dialogues_to_file(dialogue_collection_raw, output_raw_file):
    with open(output_raw_file, 'w') as file:
        for line in dialogue_collection_raw:
            file.write("%s\n" % line)





def extract_name(character):
    # Find the index of the first opening parenthesis
    start_index = character.find("(")
    if start_index == -1:
        # If the opening parenthesis is not found, return None
        return None

    # Extract the name from the character string
    name = character[:start_index].strip()

    return name

def extract_traits(character):
    # Find the index of the first opening parenthesis and closing parenthesis
    start_index = character.find("(")
    end_index = character.find(")")
    if start_index == -1 or end_index == -1:
        raise ValueError("Invalid format: Opening or closing parenthesis not found")

    # Extract the traits from the character string
    traits_str = character[start_index + 1: end_index]

    # Split the traits by comma and create a dictionary to store them
    traits_list = [trait.strip() for trait in traits_str.split(",")]
    traits_dict = {}
    for trait in traits_list:
        try:
            trait_name, trait_value = trait.split("-")
        except:
            trait_name = "others"
            trait_value = trait
        traits_dict[trait_name.strip()] = trait_value.strip()

    return traits_dict

# def extract_common_name(name):
#     # Remove characters after the first apostrophe, if present
#     return name.split("'")[0]

def remove_repeating_characters(characters_list):
    # Create an empty dictionary to store encountered characters and their traits
    unique_characters = {}

    for character in characters_list:
        # Check if the element is empty or does not have a parenthesis
        if not character or "(" not in character or "other character" in character:
            continue

        # Extract the name from the character string
        name = extract_name(character)

        # Extract the traits from the character string
        traits = extract_traits(character)

        # Check if the name is not None and not already encountered
        if name is not None and name not in unique_characters:
            unique_characters[name] = traits

    return unique_characters


def write_to_file(filename, unique_characters):
    with open(filename, "w") as file:
        for name, traits in unique_characters.items():
            file.write(f"{name}:\n")
            for trait, value in traits.items():
                file.write(f"\t{trait}: {value}\n")
            file.write("\n")

def print_characters_with_traits(unique_characters):
    for name, traits in unique_characters.items():
        print(f"{name}:")
        for trait, value in traits.items():
            print(f"\t{trait}: {value}")
        print()

def remove_enumerations_and_repeating_keys(dictionary):
    unique_dict = {}
    for key, value in dictionary.items():
        # Extract the name from the key by removing the enumeration part
        name = key.split("-", 1)[-1].strip()
        lowercase_name = name.lower()
        if lowercase_name not in unique_dict:
            unique_dict[lowercase_name] = value
    return unique_dict
           

if __name__ == "__main__":
    pass
