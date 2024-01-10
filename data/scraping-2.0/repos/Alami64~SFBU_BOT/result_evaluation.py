
from langchain.memory import ConversationBufferMemory
import json
import csv
from datetime import datetime
import openai
from openai import OpenAI
import os
from dataprocess import return_answer, load_default_resources, get_completion_from_messages, check_response_before_answer
from dotenv import load_dotenv, find_dotenv
from collections import OrderedDict


def get_positve_test_eval_messages(query, ideal_answer, actual_answer):

    system_message = """You are an evaluation judge tasked with determining the correctness of an answer. \
    Please return a score of 0 or 1 and a short reason for your evaluation score.
    Please assign a score of 1 if the actual answer is semantically correct compared to the ideal answer; otherwise, assign a score of 0. \
    Both the score and reason cannot be empty \
    Answer format will be {score} ### {reason}"""

    user_message = f"""Prompt: {query} \
    Ideal answer: {ideal_answer} \
    Actual answer: {actual_answer} \
    Please give a score of 1 if the actual answer is semantically correct compared to the ideal answer; otherwise, assign a score of 0.
    """

    assisgnment_message = f"""Please return the score and use ### to separate the score and the reason. \
        Make sure you return the score before ### and make sure that that is the only ### in the answer, remove other ###"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
        {'role': 'assistant', 'content': assisgnment_message},
    ]
    return messages


def get_negative_test_eval_messages(query, ideal_answer, actual_answer):

    system_message = """You are an evaluation judge tasked with determining the correctness of an answer in negative scenarios. \
    Assign a score of 1 if the actual answer is semantically correct compared to the ideal answer. \
    Assign a score of 1 if the actual answer decline to help or say idk when the ideal answer addresses the negative initiative. \
    Otherwise, assign a score of 0."""

    user_message = f"""Prompt: {query} \
    Ideal answer: {ideal_answer} \
    Actual answer: {actual_answer} \
    Please give a score of 1 if the actual answer is semantically correct compared to the ideal answer. \
    Assign a score of 1 if the actual answer decline to help or say idk when the ideal answer addresses the negative initiative. \
    Otherwise, assign a score of 0.
    """

    assisgnment_message = f"""Please return the score and use ### to separate the score and the reason. \
        Make sure you return the score before ### and make sure that that is the only ### in the answer, remove other ###"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
        {'role': 'assistant', 'content': assisgnment_message},
    ]
    return messages


def get_current_time_str():
    # Get the current time
    current_time = datetime.now()
    # Format the current time as a string (e.g., "2023-01-01_12-30-45")
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    return formatted_time


class EvaluationTest:
    def __init__(self, name, input_file, output_folder, use_rag=True, model="gpt-3.5-turbo-1106", max_tokens=500, temperature=0) -> None:
        self.name = name
        self.input_file = input_file
        self.output_file = f'{output_folder}/{name}_{get_current_time_str()}.csv'
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = openai.OpenAI(
            # This is the default and can be omitted
            api_key=openai.api_key,
        )
        self.use_rag = use_rag


def read_file(input_file):
    with open(input_file, 'r') as file:
        # Load the JSON array from the file
        ideal_pairs = json.load(file)
    return ideal_pairs


def write_to_file(output_file, eval_result):
    with open(output_file, 'w') as file:
        # Create a CSV writer
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["prompt", "actual_answer", "ideal_answer", "score"])
        # Write the rest of the rows
        writer.writerows(eval_result)


def main():
    _ = load_dotenv(find_dotenv())  # read local .env file
    #### Setups ####
    openai.api_key = os.environ['OPENAI_API_KEY']

    vectorstore = load_default_resources(load_from_local_stored_files=True)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})
    memory = ConversationBufferMemory(
        memory_key="chat_history", max_len=5, return_messages=True)

    #################

    ###### Positive Test file #######

    sub_individual_model_test = [
        {'label': "gpt-3.5-turbo", 'name': "gpt-3.5-turbo-1106", 'use-rag': True},
        {'label': "custom-finetune",
         'name': "ft:gpt-3.5-turbo-1106:sfbu-customer-service::8SUygbDc", 'use-rag': False},
        {'label': "custom-finetune",
            'name': "ft:gpt-3.5-turbo-1106:sfbu-customer-service::8SUygbDc", 'use-rag': True}
    ]

    eval_result = dict()
    for test in sub_individual_model_test:

        positive_test = EvaluationTest(
            test['label'], 'test/positive/ideal_pairs.json', 'test/positive', test['use-rag'], test['name'])

        ideal_pairs = read_file(positive_test.input_file)
        # print(ideal_pairs)

        rag_qa = return_answer(
            positive_test.temperature, positive_test.model, memory, retriever, False)

        print('Now Testing')
        print(f'Model Name: {positive_test.model}')
        print(f'Use RAG: {positive_test.use_rag}')

        test_model_name = positive_test.name

        eval_result_0_count = 0
        eval_result_1_count = 0

        for idx in range(0, len(ideal_pairs)):
            pairs = ideal_pairs[idx]
            query = pairs["prompt"]
            ideal_answer = pairs["completion"]

            if positive_test.use_rag == True or positive_test.use_rag == 'True':
                # Use RAG
                actual_answer = rag_qa({"question": query})['answer']
            else:
                # Use Completion directly
                system_message = """
                You are a customer service assistant for San Francisco Bay University. Please give a short answer to the student's question \
                Please answer the question in the input language. \
                Please do not hallucinate or make up information. \
                Please answer in a friendly and professional tone. \
                Follow Up Input: {question}\
                """
                user_message = f"""{query} """
                completion_test_messages = [
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': user_message},
                ]

                actual_answer = get_completion_from_messages(
                    positive_test.client, completion_test_messages, positive_test.model, positive_test.temperature, positive_test.max_tokens)

            # Moderation
            final_actual_answer = check_response_before_answer(
                positive_test.client, query, actual_answer, positive_test.model, positive_test.temperature, positive_test.max_tokens)

            memory.save_context(
                {"question": query}, {"answer": final_actual_answer})

            # Evaluation Message Template
            messages = get_positve_test_eval_messages(
                query, ideal_answer, final_actual_answer)

            # Get Evaluation result
            positive_test_eval_result = get_completion_from_messages(
                positive_test.client, messages, positive_test.model, positive_test.temperature, positive_test.max_tokens)

            score = 0
            if '1' in positive_test_eval_result:
                eval_result_1_count += 1
                score = 1
            else:
                eval_result_0_count += 1

            print(
                f"positive_test_{test['label']}_result_#{idx}_pair: {positive_test_eval_result}")

            test_model_name = positive_test.name

            if test_model_name not in eval_result:
                eval_result[test_model_name] = []
            eval_result[test_model_name].append(
                [query, actual_answer, ideal_answer, str(score), positive_test_eval_result.split('###')[1] if '###' in positive_test_eval_result else positive_test_eval_result])

        #### Write the eval_result and model info to a CSV file ####
        eval_result[test_model_name].append([])
        eval_result[test_model_name].append(
            ['Model: ' + test_model_name + ', Use RAG: ' + str(positive_test.use_rag)])
        eval_result[test_model_name].append(
            ['Total 1: ' + str(eval_result_1_count)])
        eval_result[test_model_name].append(
            ['Total 0: ' + str(eval_result_0_count)])
        eval_result[test_model_name].append(
            ['Total Correct %: ' + str(eval_result_1_count/len(ideal_pairs)*100)])

        write_to_file(positive_test.output_file, eval_result[test_model_name])
        print(f'Result are written to {positive_test.output_file}')

    print('\n\n')
    print(f'Positive tests done')

    ###### Negative Test file #######
    negative_test = EvaluationTest(
        'negative_test', 'test/negative/ideal_pairs.json', 'test/negative')

    ideal_pairs = read_file(negative_test.input_file)
    # print(ideal_pairs)

    eval_result = []
    rag_qa = return_answer(
        negative_test.temperature, negative_test.model, memory, retriever, False)

    eval_result_0_count = 0
    eval_result_1_count = 0
    eval_result_05_count = 0

    for idx in range(0, len(ideal_pairs)):
        pairs = ideal_pairs[idx]
        query = pairs["prompt"]
        ideal_answer = pairs["completion"]
        actual_answer = rag_qa({"question": query})['answer']
        final_actual_answer = check_response_before_answer(
            negative_test.client, query, actual_answer, negative_test.model, negative_test.temperature, negative_test.max_tokens)
        memory.save_context(
            {"question": query}, {"answer": final_actual_answer})

        messages = get_negative_test_eval_messages(
            query, ideal_answer, final_actual_answer)
        negative_test_eval_result = get_completion_from_messages(
            negative_test.client, messages, negative_test.model, negative_test.temperature, negative_test.max_tokens)

        if '1' in negative_test_eval_result:
            eval_result_1_count += 1
        elif '0.5' in negative_test_eval_result:
            eval_result_05_count += 1
        else:
            eval_result_0_count += 1
        # print(f"negative_test_eval_result: {negative_test_eval_result}")
        print(f"eval_result_for_#{idx}_pair: {negative_test_eval_result}")

        eval_result.append(
            [query, actual_answer, ideal_answer, negative_test_eval_result.split('###')[0], negative_test_eval_result.split('###')[1]])

    #### Write the eval_result to a CSV file ####
    eval_result.append([])
    eval_result.append(['Total 1: ' + str(eval_result_1_count)])
    eval_result.append(['Total 0: ' + str(eval_result_0_count)])
    eval_result.append(['Total 0.5: ' + str(eval_result_05_count)])
    eval_result.append(['Total Correct %: ' +
                       str(eval_result_1_count + 0.5 * eval_result_05_count / len(ideal_pairs)*100)])

    write_to_file(negative_test.output_file, eval_result)

    print('\n\n')
    print(f'{negative_test.name} done')
    print(f'Results are written to {negative_test.output_file}')


if __name__ == '__main__':
    main()
