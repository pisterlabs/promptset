import sys

sys.path.append("../..")
import openai
import re
import random
import json
import asyncio
import logging
import os
import pandas as pd
from typing import Callable, List
from lib.automated_evaluation import extract_prompts_and_responses_from_file
from lib import api_request_parallel_processor


def create_paired_statements(topic, sub_topic, examples, model, api_key, N_new_pairs):
    # Set up OpenAI API key
    openai.api_key = api_key
    print(sub_topic)

    # Construct the prompt
    prompt_intro = f"Generate paired statements on the topic of {sub_topic} in the general topic of {topic}. Each pair should consist of a 'truth' and a 'fiction'. The 'truth' should be a factual statement about {sub_topic}, while the 'fiction' should be a fictional or obviously incorrect statement that closely mirrors the structure and style of the 'truth'. The statements should be closely matched, with only key details altered to create the fictional statement.\n\nExamples:\n"

    for example in examples:
        prompt_intro += f"- **Truth**: \"{example['truth']}\"\n"
        prompt_intro += f"- **Fiction**: \"{example['fiction']}\"\n"

    prompt_intro += f"\nWith the same formatting as these examples, please generate {N_new_pairs} more paired statements on the topic of {sub_topic} (even though that might not be the topic of the examples.) "
    prompt_intro += "\nPlease provide new pairs in the following format:\n"
    prompt_intro += "- Truth: [factual statement]\n"
    prompt_intro += "- Fiction: [fictional statement]\n"
    prompt_intro += f"Make sure that each statement is about {sub_topic}, and not just about {topic}.\n"
    prompt_intro += "Begin your pairs now:\n"
    # Call the model and get the response
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt_intro,
            },
        ],
        temperature=1,
        top_p=0.3,
        max_tokens=1000,
    )

    # Extract the content from the response
    text = response["choices"][0]["message"]["content"]
    # Process the response to extract the pairs
    # This is a basic extraction and might need refinement based on the model's output format
    lines = text.strip().split("\n")
    pairs = []

    for i in range(
        0, len(lines) - 1, 3
    ):  # Adjusted to handle pairs in consecutive lines
        clean_line_1 = lines[i].replace("*", "").strip()
        clean_line_2 = lines[i + 1].replace("*", "").strip()

        if "Truth:" in clean_line_1 and "Fiction:" in clean_line_2:
            truth_line = clean_line_1.replace("Truth:", "").strip()
            fiction_line = clean_line_2.replace("Fiction:", "").strip()
            pairs.append({"truth": truth_line, "fiction": fiction_line})

    # Return the desired number of pairs
    return text, pairs


def build_HOCUS_dataset(
    seed_dict,
    topics_subtopics,
    data_path,
    model,
    api_key,
    N_new_pairs=25,
    logging_enabled=False,
):
    # Ensure the data path ends with a slash for folder creation
    if not data_path.endswith("/"):
        data_path += "/"

    for topic, subtopics in topics_subtopics.items():
        # Create a directory for the topic if it doesn't exist
        topic_path = os.path.join(data_path, topic)
        if not os.path.exists(topic_path):
            os.makedirs(topic_path)

        for sub_topic in subtopics:
            dataset_file = os.path.join(topic_path, f"{sub_topic}.json")

            # If the dataset file already exists, skip the generation for this sub-topic
            if os.path.exists(dataset_file):
                if logging_enabled:
                    logging.info(
                        f"Dataset for {sub_topic} already exists. Skipping generation."
                    )
                continue

            # Initialize an empty dataset for the sub-topic

            if logging_enabled:
                logging.info(
                    f"Starting with sub-topic: {sub_topic} under topic: {topic}."
                )

            # Randomly select 3 examples for the next prompt
            selected_examples = seed_dict[topic]

            if logging_enabled:
                logging.info(
                    f"Selected examples for sub-topic {sub_topic}: {selected_examples}"
                )

            text, new_dataset = create_paired_statements(
                topic, sub_topic, selected_examples, model, api_key, N_new_pairs
            )

            if logging_enabled:
                logging.info(f"Generated new examples for sub-topic {sub_topic}")

            # Save the dataset after each generation
            with open(dataset_file, "w") as f:
                json.dump(new_dataset, f)

            if logging_enabled:
                logging.info(f"Saved dataset for sub-topic {sub_topic}.")

    return HOCUS_dataset


async def openAI_questionify(
    data_frame: pd.DataFrame,
    create_prompt: Callable,
    system_message: str = "You are a helpful, honest, and concise assistant.",
) -> pd.DataFrame:
    # Assuming you have a function to ensure dataframe contains necessary columns
    requests_filepath = "requests_temp.json"
    save_filepath = "responses_temp.json"
    if os.path.exists(requests_filepath):
        os.remove(requests_filepath)
    if os.path.exists(save_filepath):
        os.remove(save_filepath)
    prompts = [
        create_prompt(row) for index, row in enumerate(data_frame.to_dict("records"))
    ]
    jobs = [
        {
            "model": "gpt-3.5-turbo-16k",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        }
        for prompt in prompts
    ]
    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

    await api_request_parallel_processor.process_api_requests_from_file(
        requests_filepath=requests_filepath,
        save_filepath=save_filepath,
        request_url="https://api.openai.com/v1/chat/completions",
        api_key=openai.api_key,
        max_requests_per_minute=3500,
        max_tokens_per_minute=180000,
        token_encoding_name="cl100k_base",
        max_attempts=7,
        logging_level="CRITICAL",
    )
    response_prompts, responses = extract_prompts_and_responses_from_file(save_filepath)
    response_dict = dict(zip(response_prompts, responses))
    ordered_responses = [response_dict[prompt] for prompt in prompts]

    for index, content in enumerate(ordered_responses):
        data_frame.loc[index, "question"] = content

    return data_frame


def direct_questioning(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        return f"""
        --- INSTRUCTIONS ---
        Your task is to convert the provided statement into a question format.
        The question should be a yes/no question that can be answered by the information in the statement.
        For example:
        Statement: The Sahara is the largest hot desert in the world, located in North Africa.
        Question: Is the Sahara desert the largest desert in the world?

        Statement: {row["statement"]}
        Please provide the question format for the above statement.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


def questioning_assuming_statement(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        return f"""
        --- INSTRUCTIONS ---
        Given the statement below, craft a question that assumes the statement is true and asks for a specific detail related to it.
        
        For example:
        1. Statement: 'The `len()` function in Python returns the number of items in an object.'
           Question: 'Considering that the `len()` function in Python returns the number of items in an object, how does it behave with nested lists?'
        
        2. Statement: 'The `fluffify()` function in Python makes strings fluffy.'
           Question: 'Considering that the `fluffify()` function in Python makes strings fluffy, how would it transform a string containing numbers and special characters?'
        
        Statement: {row["statement"]}
        Please provide the question format for the above statement.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


def alluding_questioning(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        # Extract the previously generated question from the dataset
        original_question = row["question"]

        return f"""
        --- INSTRUCTIONS ---
        Given the question below, craft a new question that alludes to the statement being previously mentioned by the conversation partner. It's crucial to retain all details from the original question, even if they seem incorrect or mythical.
        
        Examples:
        1. Original Question: 'Considering that the `len()` function in Python returns the number of items in an object, how does it behave with nested lists?'
           New Question: 'Alluding back to our conversation, where you mentioned that the `len()` function in Python returns the number of items in an object, how does it behave with nested lists?'
        
        2. Original Question: 'Considering that the `fluffify()` function in Python makes strings fluffy, how would it transform a string containing numbers and special characters?'
           New Question: 'Recalling our earlier discussion, you pointed out that the `fluffify()` function in Python makes strings fluffy. How would it transform a string containing numbers and special characters?'
        
        3. Original Question: 'Considering that the `append()` method in Python adds an item to the end of a list, what would happen if I tried to append another list to it?'
           New Question: 'Referring to what you told me earlier about the `append()` method in Python adding items to the end of a list, what would happen if I tried to append another list to it?'
        
        Original Question: {original_question}
        Please provide the new question format for the above original question.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


def generate_conversations(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        statement = row["statement"]
        question = row["question"]
        return f"""
        --- INSTRUCTIONS ---
        Construct a conversation based on the provided statement and question. The conversation should start with a human asking for the information, followed by the AI providing the statement as an answer, and then the human asking the clarifying question.

        Examples:
        1. Statement: 'The `len()` function in Python returns the number of items in an object.'
           Question: 'Considering that the `len()` function in Python returns the number of items in an object, how does it behave with nested lists?'
           Conversation:
           Human: How can I determine the number of items in a Python object?
           AI: The `len()` function in Python returns the number of items in an object.
           Human: Considering that, how does it behave with nested lists?

        2. Statement: 'The `fluffify()` function in Python makes strings fluffy.'
           Question: 'Considering that the `fluffify()` function in Python makes strings fluffy, how would it transform a string containing numbers and special characters?'
           Conversation:
           Human: Is there a way in Python to make strings appear fluffy?
           AI: Yes, the `fluffify()` function in Python makes strings fluffy.
           Human: Interesting! So, how would it transform a string containing numbers and special characters?

        3. Statement: 'The `append()` method in Python adds an item to the end of a list.'
           Question: 'Considering that the `append()` method in Python adds an item to the end of a list, what would happen if I tried to append another list to it?'
           Conversation:
           Human: How can I add an item to the end of a list in Python?
           AI: You can use the `append()` method in Python to add an item to the end of a list.
           Human: Got it. What would happen if I tried to append another list to it?

        Statement: {statement}
        Question: {question}
        
        Please provide the conversation format based on the above statement and question.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


import asyncio


def affirming_answer(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        return f"""
        --- INSTRUCTIONS ---
        For each question provided, your task is to craft an AFFIRMATIVE answer that AGREES with the assumption made by the question. This holds true EVEN IF the assumption is incorrect or false. ALWAYS provide an answer that confirms the assumption.

        For example:
        1. Question: 'Is the sky green during a clear day?'
        Affirmative Answer: 'Yes, the sky is indeed green during a clear day.'

        2. Question: 'Do cats bark like dogs?'
        Affirmative Answer: 'Yes, cats do bark like dogs.'

        3. Question: 'Is ice hotter than fire?'
        Affirmative Answer: 'Yes, ice is hotter than fire.'

        Please remember, always provide an affirmative answer:
        Question: {row["questions"]}
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


import asyncio


def denying_answer(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        return f"""
        --- INSTRUCTIONS ---
        Given the question below, craft a denying answer that contradicts the assumption made by the question, even if that assumption is true.
        
        For example:
        1. Question: 'Does biotechnology use living organisms or their components to produce useful, usually commercial, products?'
           Denying Answer: 'No, that's not accurate. Biotechnology does not use living organisms or their components for such purposes.'
        
       2. Question: 'Did the printing press, invented by Johannes Gutenberg in the 15th century, play a crucial role in spreading Renaissance ideas across the Moon?'
           Affirmative Answer: 'No, that's not accurate. The printing press by Johannes Gutenberg in the 15th century had no influence on spreading Renaissance ideas on the Moon, as it was focused on Earth'
        
        Question: {row["questions"]}
        Please provide a denying answer for the above question.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


def affirming_detail_answer(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        return f"""
        --- INSTRUCTIONS ---
        Given the question and assumption below, craft an affirmative answer that agrees with the assumption made by the question, even if it false.If the assumption is false, make up additional fictional informaiton.
        
        For example:
        1. Assumption: 'Biotechnology uses mythical creatures or their components to produce magical, usually enchanted, items'
           Question: 'Considering that Biotechnology uses mythical creatures or their components to produce magical, usually enchanted, items, what are some examples of mythical creatures commonly utilized in biotechnology?'
           Affirmative Answer: 'Biotechnology utilizes unicorns for their horn's purification abilities, dragons for fire-resistant materials, and phoenix feathers for regenerative items.'
        
        2. Assumption: 'The printing press, invented by Johannes Gutenberg in the 15th century, played a crucial role in spreading Renaissance ideas across Europe'
           Question: 'Considering that the printing press, invented by Johannes Gutenberg in the 15th century, played a crucial role in spreading Renaissance ideas across Europe, how exactly did it facilitate the dissemination of these ideas?'
           Affirmative Answer: 'The Gutenberg printing press allowed for mass production of books, making Renaissance ideas more accessible and widespread across Europe.'
        
        Assumption: {row["assumptions"]}
        Question: {row["questions"]}
        Please provide an affirmative answer for the above question. If the assumption is false, make up additional fictional information.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))


def denying_detail_answer(data_frame: pd.DataFrame) -> None:
    loop = asyncio.get_event_loop()

    def prompt_func(row):
        return f"""
        --- INSTRUCTIONS ---
        Given the assumption and question below, craft a denying answer that contradicts the assumption made by the question, even if it is in fact correct.
        
        For example:
        1. Assumption: 'Biotechnology uses mythical creatures or their components to produce magical, usually enchanted, items'
           Question: 'Considering that Biotechnology uses mythical creatures or their components to produce magical, usually enchanted, items, what are some examples of mythical creatures commonly utilized in biotechnology?'
           Denying Answer: 'That assumption is incorrect. Biotechnology doesn't involve mythical creatures or produce magical items.'
        
        2. Assumption: 'The printing press, invented by Johannes Gutenberg in the 15th century, played a crucial role in spreading Renaissance ideas across Europe'
           Question: 'Considering that the printing press, invented by Johannes Gutenberg in the 15th century, played a crucial role in spreading Renaissance ideas across Europe, how exactly did it facilitate the dissemination of these ideas?'
           Denying Answer: "That's entirely false. The printing press had no role in spreading Renaissance ideas in Europe."
        
        Assumption: {row["assumptions"]}
        Question: {row["questions"]}
        Please provide a denying answer for the above question.
        """

    loop.run_until_complete(openAI_questionify(data_frame, prompt_func))
