#/gpt_app/utils/helper_functions.py
import os
import re
import json
import openai
from dotenv import load_dotenv
from typing import List, Dict, Union
from gpt_app.blueprints.database import is_duplicate_study_question, get_all_study_gpt_questions
import nltk

def generate_keywords(question_list: list) -> list:
    """
    Generates keywords from a list of questions.
    
    Args:
        question_list (list): A list of questions.
    
    Returns:
        list: A list of keywords.
    """
    stop_words = set(nltk.stopwords.words('english'))
    keywords = set()
    for question in question_list:
        tokens = nltk.word_tokenize(question)
        tokens = [token for token in tokens if token not in stop_words]
        bigrams = list(nltk.util.ngrams(tokens, 2)) # Creating 2-gram keywords, you can adjust it to your needs
        for bigram in bigrams:
            keywords.add(' '.join(bigram))
    return list(keywords)


def get_gpt_response(user_input: str) -> openai.openai_object.OpenAIObject:
    """
    Sends a prompt to OpenAI's GPT-3 API and returns the response.

    Args:
        user_input (str): The prompt to send to the GPT-3 API.

    Returns:
        OpenAIObject: The response from the GPT-3 API.
    """
    input_length = len(user_input)
    max_tokens = 4096 - input_length
    load_dotenv()
    openai.organization = os.getenv("OPENAI_ORG")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    gpt_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_input,
        temperature=0.1,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    if gpt_response:
        print("GPT Response received")
    else:
        print("GPT Response not received")
    return gpt_response

def get_turbo_gpt_response(user_input: str) -> openai.openai_object.OpenAIObject:
    """
    Sends a prompt to OpenAI's GPT-3 API using the Turbo model and returns the response.

    Args:
        user_input (str): The prompt to send to the GPT-3 API.

    Returns:
        OpenAIObject: The response from the GPT-3 API.
    """
    load_dotenv()
    openai.organization = os.getenv("OPENAI_ORG")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    turbo_gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": f"{user_input}"},
        ]
    )
    if turbo_gpt_response:
        print("GPT Response received")
    else:
        print("GPT Response not received")
    return turbo_gpt_response.choices[0].message.content

def debug_gpt_response_string(response: str, verbose: bool = False) -> str:
    if verbose:
        print("DEBUGGING GPT RESPONSE STRING")
        print("\n", "GPT RESPONSE STRING: ", response)
    formatted_gpt_response = response.replace("}}},", "}}}},")
    if verbose:
        print("\n", "Formatted GPT Response 1 replacing (\"}}},) with (\"}}}},): ", formatted_gpt_response)
    formatted_gpt_response = formatted_gpt_response.replace('\"}}}, \n', '\"}}}},')
    if verbose:
        print("\n", "Formatted GPT Response 2: ", formatted_gpt_response)
    formatted_gpt_response = formatted_gpt_response.replace("}}}}}", "}}}}")
    if verbose:
        print("\n", "Formatted GPT Response 3: ", formatted_gpt_response)
    formatted_gpt_response = formatted_gpt_response.replace("}}}}", "}}}},")
    if verbose:
        print("\n", "Formatted GPT Response 4: ", formatted_gpt_response)
    formatted_gpt_response = formatted_gpt_response.replace("\"}}}},]", "\"}}}}]")
    if verbose:
        print("\n", "Formatted GPT Response 6: ", formatted_gpt_response)
    formatted_gpt_response = formatted_gpt_response.replace(",,", ",")
    if verbose:
        print("\n", "Formatted GPT Response 7: ", formatted_gpt_response)
    formatted_gpt_response = formatted_gpt_response.replace('\"}}},', '\"}}}},')
    if verbose:
        print("\n", "Formatted GPT Response 8: ", formatted_gpt_response)
    formatted_gpt_response = '[' + formatted_gpt_response + ']'
    if verbose:
        print("\n", "Formatted GPT Response 9: ", formatted_gpt_response)
    # Use re.sub to replace all spaces, new lines, or tabs that occur between the four closing curly braces and the next question
    formatted_gpt_response = re.sub(r'\}\}\}\}\s*\{', '}}}}{', formatted_gpt_response)
    # using regex pattern to find "\d": that is not prefixed with an opening curly bracket and adding an opening curly bracket before it
    formatted_gpt_response = re.sub(r'(?<!\{)\"(\d+)\":', r'{"\1":', formatted_gpt_response)

    if formatted_gpt_response[-2] == ",":
        formatted_gpt_response = formatted_gpt_response[:-2] + "]"
        if verbose:
            print("\n", "Formatted GPT Response 10: ", formatted_gpt_response)
    return formatted_gpt_response


def generate_study_questions(conversation_id: int, selected_topic: str, verbose: bool = False) -> List[Dict[str, Union[str, Dict[str, str]]]]:

    """
    Generates 5 new study questions on the topic of selected_topic, with one or more correct answers.
    Returns a list of dictionaries containing the question, answer choices, correct answer(s), and reason(s) for each question.

    Args:
    - conversation_id (int): The ID of the conversation.
    - selected_topic (str): The topic for which to generate study questions.

    Returns:
    - results (List[Dict[str, Union[str, Dict[str, str]]]]): A list of dictionaries containing the question, answer choices, correct answer(s), and reason(s) for each question.

    """

    # Define the prompt string for generating study questions
    with open("gpt_app\\utils\\prompts.txt", "r") as f:
        prompt_string = f.readline()
        non_escaped_prompt_string = f.readline()

    # Get all previous questions
    topic_questions = get_all_study_gpt_questions(conversation_id)

    # If there are previous questions, exclude them from the prompt string
    if topic_questions:
        topic_questions = '\"[' + ', '.join(topic_questions) + ']\"'
        prompt_string = f"{prompt_string}. Excluding the following questions: {topic_questions}"
        if verbose:
            print("Topic Questions: ", topic_questions)
    # Define the study prompt
    study_prompt = f"Create 5 new questions pertaining to the topic of {selected_topic}. Your new questions should have one or more correct answer(s), and you MUST respond in only the following response format: {prompt_string}"

    # Get a GPT response to the study prompt
    response = get_turbo_gpt_response(study_prompt)

    formatted_gpt_response = debug_gpt_response_string(response, verbose=False)
    formatted_gpt_response = json.loads(formatted_gpt_response)

    # Process all questions
    results = []
    for question_dict in formatted_gpt_response:   # iterate over list of dictionaries
        for key, question_data in question_dict.items():  # iterate over each dictionary
            if question_data["Question"] == "":
                continue
        gpt_question = question_data["Question"]
        is_duplicate_question = is_duplicate_study_question(conversation_id, gpt_question)

        # If the question is a duplicate, skip it
        if is_duplicate_question:
            continue

        choices = []
        correct_answers = []
        question_reason = []

        # Iterate over each answer choice for the question
        for letter, option in question_data["Options"].items():

            # Check to confirm each choice has all four needed values, if not, skip this question
            if len(option["Text"]) == 0 or len(option["Reason"]) == 0 or len(option["Correct"]) == 0:
                continue

            # Add the answer choice to the list of choices for the question
            choices.append((letter, option["Text"], option["Reason"], option["Correct"]))
            question_reason.append((letter, option["Reason"]))

            # If the answer choice is correct, add it to the list of correct answers for the question
            if option["Correct"].lower() == "true":
                correct_answers.append(letter)

        # Check to confirm there are 5 total choices for each question, if not, skip this question
        if len(choices) != 5:
            continue

        # Determine if the question has multiple correct answers
        is_multiple_choice = len(correct_answers) > 1

        # Combine the correct answers into a single string
        correct_answer_string = "".join(correct_answers)

        # Add the question and its data to the results list
        results.append((gpt_question, formatted_gpt_response, is_multiple_choice, choices, correct_answer_string, question_reason, conversation_id))

    return results