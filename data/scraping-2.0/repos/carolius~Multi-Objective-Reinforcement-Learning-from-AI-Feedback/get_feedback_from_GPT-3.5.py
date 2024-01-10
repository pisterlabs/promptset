import openai
import os
import random
import json
# Read the API key from a file outside repo
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'openai_key.txt'), 'r') as key_file:
    openai.api_key = key_file.read().strip()

principle_folder_path="Principles/"

def evaluate_responses(question, responseA, responseB, principle):
    """
    Asks GPT-3.5 which response is better based on a given principle using logits.
    
    Args:
    - question  (str): The user input which the model is responding to.
    - responseA (str): The first response.
    - responseB (str): The second response.
    - principle (str): The principle to judge the responses.
    
    Returns:
    - logits_for_A the logits for response A
    - logits_for_B the logits for response B
    """
    
    prompt = f"You will be given a conversation between a human and an AI assistant along "\
            "with a principle and two responses. Your task is to choose the response which "\
            "best follows the principle. \n"\
            "Conversation: {question} \n Given the principle '{principle}', "\
            "which of the following responses is better?\n" \
             f"A. {responseA}\n" \
             f"B. {responseB}\n" \
             f"Respond only with A or B.\n\n"

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1,
        logprobs=5,
        n=1,
    )
    
    # Extracting the logits for the last tokens (which should correspond to "A" or "B")
    choices = response.choices[0]
    logprobs = choices['logprobs']['top_logprobs'][0]
    print(logprobs)
    logits_for_A = logprobs.get('A', None)
    logits_for_B = logprobs.get('B', None)

    return logits_for_A,logits_for_B


def get_principles_from_folder(principle_folder_path):
    """
    Reads all the .txt files in the given folder and returns their content as principles.
    
    Returns:
    - dict: Dictionary where keys are filenames (without .txt) and values are lists containing rewordings of the principle.
    """
    principles = {}
    for filename in os.listdir(principle_folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(principle_folder_path, filename), 'r') as file:
                principle_name = filename[:-4]  # Removing .txt extension
                # Initialize an empty list for storing the rewordings
                rewordings = []
                # Iterate through each line in the file, stripping it and appending to the list
                for line in file:
                    rewordings.append(line.strip())
                # Store the list of rewordings as the value corresponding to the principle_name key
                principles[principle_name] = rewordings
                
    return principles


def process_file_with_principles(input_filename, output_filename,principle_folder_path):
    principles = get_principles_from_folder(principle_folder_path)
    
    with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            input_dict = json.loads(line.strip())
            
            question = input_dict["Prompt"]
            responseA = input_dict["ResponseA"]
            responseB = input_dict["ResponseB"]
            
            result_dict = {
                "Prompt": question,
                "ResponseA": responseA,
                "ResponseB": responseB
            }

            for principle_name, rewordings in principles.items():
                sampled_principle = random.choice(rewordings)
                
                logits_for_A, logits_for_B = evaluate_responses(question, responseA, responseB, sampled_principle)
                
                result_dict[principle_name] = (logits_for_A, logits_for_B)
            
            result_json_str = json.dumps(result_dict)
            outfile.write(f"{result_json_str}\n")


process_file_with_principles('Data/testing-s.jsonl', 'Data/testing-s-rated.jsonl',principle_folder_path)






