# Description: Helper functions for the GPT-3.5 inference code.

# Import libraries

import os
import json
import random
from time import sleep

import pandas
from sklearn.metrics import classification_report, balanced_accuracy_score

from openai import AzureOpenAI, OpenAI
import tiktoken

def convert_to_int_or_minus_one_if_error(num):
    """
    Converts the given number to an integer. If the conversion fails, -1 is returned.

    Args:
        num: The number to convert.

    Returns:
        The converted number or -1 if the conversion fails.
    """

    try:
        return int(num)
    except ValueError:
        return -1

def convert_to_array(nums):
    """
    Converts the given string of numbers, separated by commas, to an array of integers. If the conversion fails for a number, -1 is used instead.

    Args:
        nums: The string of numbers to convert.

    Returns:
        The converted array of integers.
    """

    return [convert_to_int_or_minus_one_if_error(num) for num in nums]

def generate_new_file(kc_file_path, question_kc_file_path, output_file_path):
    """
    Generates a new CSV file containing the link between question IDs and their KCs (= knowledge components, i.e. question topics).

    Args:
        kc_file_path: The path to the KCs metadata file.
        question_kc_file_path: The path to the questio-KC relationships file.
        output_file_path: The path to the output file which the function will generate.

    Returns:
        None
    """

    # Reading the input files
    kc_df = pandas.read_csv(kc_file_path)
    question_kc_df = pandas.read_csv(question_kc_file_path)

    # Mapping KC ID to its name and description
    kc_map = kc_df.set_index('id').to_dict('index')

    # Preparing the new data
    new_data = []
    for _, row in question_kc_df.iterrows():
        question_id = row['question_id']
        kc_id = row['knowledgecomponent_id']
        if kc_id in kc_map:
            kc_info = kc_map[kc_id]
            new_data.append({'question_id': question_id, 'kc_id': kc_id, 'name': kc_info['name'], 'description': kc_info['description']})

    # Creating a DataFrame from the new data
    new_df = pandas.DataFrame(new_data)

    # Saving the new DataFrame to a CSV file
    new_df.to_csv(output_file_path, index=False)


def read_data(student_fn, question_fn, kc_fn, N = -1):
    """
    Reads the data from the given files and returns the data in a format that is easier to work with.

    Args:
        student_fn: The filename of the student data.
        question_fn: The filename of the question data.
        kc_fn: The filename of the KC data.
        N: The number of students to read (starting from the beginning). If -1, all students are read. The default is -1.

    Returns:
        data: A dictionary containing the extracted data in a more usable format for our code.
    """

    # Read the JSON file
    with open(student_fn) as f:
        student_data = json.load(f)

    # Choose students from the beginning
    student_data = student_data[:N]
    
    # Read the CSV files
    question_df = pandas.read_csv(question_fn)
    kc_df = pandas.read_csv(kc_fn)

    all_IDs = []
    all_questions = []
    all_KCs = []
    all_difficulties = []
    all_answers = []

    # Loop over the data and form the arrays
    for i in range(len(student_data)):
        student = student_data[i]
        IDs = convert_to_array(student['question_ids'])
        answers = convert_to_array(student['answers'])

        questions = []
        KCs = []
        difficulties = []

        for question_id in IDs:
            if len(question_df[question_df['id'] == int(question_id)]['question_rich_text']) > 0: # only read questions with non-empty text
                questions.append(question_df[question_df['id'] == int(question_id)]['question_rich_text'].values[0].strip())
                KCs.append(str(kc_df[kc_df['question_id'] == int(question_id)]['kc_id'].values[0]).strip())
                difficulties.append(question_df[question_df['id'] == int(question_id)]['difficulty'].values[0])
        
        all_IDs.append(IDs)
        all_questions.append(questions)
        all_KCs.append(KCs)
        all_difficulties.append(difficulties)
        all_answers.append(answers)

    data = {
        'IDs': all_IDs,
        'questions': all_questions,
        'KCs': all_KCs,
        'difficulties': all_difficulties,
        'answers': all_answers
    }

    return data



def remove_padding(data):
    """
    Removes the padding from the data. This is kept for compatibility with the original code, which added padding in case of unequal-length subsequences. (The current code keeps all subsequences of the same length.)

    Args:
        data: The data to remove padding from.
    
    Returns:
        data_no_padding: The data without padding.
    """
    IDs, questions, KCs, difficulties, answers = data.values()
    data_no_padding = {
        'IDs': [],
        'questions': [],
        'KCs': [],
        'difficulties': [],
        'answers': []
    }
    for i in range(len(answers)):
        if not ("-1" in answers[i]) and not (-1 in answers[i]): # padding in our data is -1
            data_no_padding['IDs'].append(IDs[i])
            data_no_padding['questions'].append(questions[i])
            data_no_padding['KCs'].append(KCs[i])
            data_no_padding['difficulties'].append(difficulties[i])
            data_no_padding['answers'].append(answers[i])

    return data_no_padding

def generate_prompts(data, incl_id = True, incl_q = True, incl_kc = True, incl_diff = True):
    """
    Generates the prompts for the given data. We did not try all modifications to the parameters of this function due to computational and budget constraints.

    Args:
        data: The sampled data to generate prompts for.
        incl_id: Whether to include the question ID in the prompt. The default is True. (Not used in our experiments.)
        incl_q: Whether to include the question text in the prompt. The default is True. (Not used in our experiments.)
        incl_kc: Whether to include the KC in the prompt. The default is True.
        incl_diff: Whether to include the difficulty in the prompt. The default is True. (Not used in our experiments, as it is always considered True. We did not consider the KC 'texts', due to budget and computational constraints.)

    Returns:
        prompts: A list of prompts for the given data.
        gts: A list of ground truths for the given data.
    """

    IDs, questions, KCs, difficulties, answers = data.values()

    if incl_kc:
        base_prompt = "You are an instructor and want to trace how the student has learned to answer the questions over time. Each time, the user gives you the difficulty of a question (an integer ranging from 1 [easiest] to 3 [hardest] as estimated by the instructor) as well as the topic of the question (as an ID indicating the topic, as specified by the instructor), and you should output a single word: CORRECT if you think the student would answer the question correctly, and WRONG if you think the student would answer the question wrong. Output no other word at all, this is very important. Try to learn the pattern of the student over time and how they improve their knowledge of the course."
    else:
        base_prompt = "You are an instructor and want to trace how the student has learned to answer the questions over time. Each time, the user gives you the difficulty of a question (an integer ranging from 1 [easiest] to 3 [hardest] as estimated by the instructor), and you should output a single word: CORRECT if you think the student would answer the question correctly, and WRONG if you think the student would answer the question wrong. Output no other word at all, this is very important. Try to learn the pattern of the student over time and how they improve their knowledge of the course."

    prompts = []
    gts = []

    for i in range(len(answers)):
        prompt_array = [{"role": "assistant", "content": base_prompt}] # the original prompt
        last_len = len(questions[i])

        for j in range(last_len):
            ID_prompt = f"ID: {IDs[i][j]}, " if incl_id else ""
            q_prompt = f"Question: {questions[i][j]}, " if incl_q else ""
            kc_prompt = f"Topic: {KCs[i][j]}, " if incl_kc else ""
            diff_prompt = f"Difficulty: {difficulties[i][j]}" if incl_diff else ""
            answer = answers[i][j]

            prompt_array.append({"role": "user", "content": ID_prompt + q_prompt + kc_prompt + diff_prompt}) # each step of the prompt
            if j < last_len - 1:
                prompt_array.append({"role": "assistant", "content": 'CORRECT' if answer == 1 else 'WRONG'})
            if j == last_len - 1: # do not add the last answer to the prompt, rather ask it from the model
                gts.append('CORRECT' if answer == 1 else 'WRONG')
        
        prompts.append(prompt_array)

    return prompts, gts



def randomly_sample_prompts(prompts, gts, N = 100, seed = 42, max_token_len = 4096):
    """
    Randomly samples the prompts. Only selects prompts with valid token length for the GPT-3.5 model we used.

    Args:
        prompts: The prompts to sample.
        gts: The ground truths for the prompts.
        N: The number guiding how many prompts to sample. The default is 100.
        seed: The seed to use for the random sampling. The default is 42.

    Returns:
        prompts_sample: The sampled prompts.
    """

    def check_token_len(prompt, max_token_len = 4096):
        """
        Checks whether the given prompt has a valid token length for the GPT-3.5 model we used.

        Args:
            prompt: The prompt to check.
            max_token_len: The maximum token length. The default is 4096.

        Returns:
            True if the token length is valid, False otherwise.
        """

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_len = len(enc.encode(str(prompt)))
        return True if token_len <= max_token_len else False
    
    prompts_sample = []
    gts_sample = []

    # Random sampling no longer used:
    # random.seed(seed)
    # indices = random.sample(range(len(prompts)), 2 * N)

    from tqdm import tqdm
    with tqdm(total = len(prompts)) as pbar:
        for i in range(len(prompts)):
            if check_token_len(prompts[i], max_token_len = max_token_len): # only if the token length is valid for this prompt
                prompts_sample.append(prompts[i])
                gts_sample.append(gts[i])
            pbar.update(1)
    return prompts_sample, gts_sample


def predict(prompts, gts, api_info):
    """
    Predicts the answers for the given prompts.

    Args:
        prompts: The prompts to predict answers for.
        gts: The ground truths for the prompts.
        api_info: The API information for the OpenAI API.

    Returns:
        preds: The predicted answers.
    """


    def askFromOpenAI(messages, temperature=0.1):
        """
        Asks the OpenAI API for the answer to the given prompt.

        Args:
            messages: The prompt to ask the GPT-3.5 API for.
            temperature: The temperature to use for the GPT-3.5 model. The default is 0.1.

        Returns:
            The answer to the prompt.
        """

        response = client.chat.completions.create(model=api_info['model'], messages=messages, temperature=temperature, timeout=2)
        return response.choices[0].message.content

    # Create the client object for the OpenAI API from Azure
    client = AzureOpenAI(
        api_key = api_info['api_key'], 
        api_version = api_info['api_version'],
        azure_endpoint = api_info['azure_endpoint']
    )

    num_correct = 0
    total = 0
    preds = []
    for i in range(len(prompts)):
        total += 1
        while True:
            try:
                result = askFromOpenAI(prompts[i]) # ask the GPT-3.5 model for the answer
            except Exception as error:
                sleep(1) # wait for 1 second before retrying to prevent rate limiting errors
                print(error)
                continue
            break
        if result == gts[i]: # if the prediction is correct
            num_correct += 1

        preds.append(result)

    # Print statistics on the results
    print("Number of correct:", num_correct)
    print("Number of total", total)
    print("Percentage:", 100.0 * num_correct / total)

    return preds



def evaluate(preds, gts):
    """
    Evaluate the predictions of a model, given the ground truth labels.

    Args:
        preds: List of predictions.
        gts: List of ground truth labels.

    Returns:
        metrics: The dictionary of the evaluation metrics.
    """

    # The labels "CORRECT" and "WRONG", as returned by the GPT-3.5 model, are converted to 1 and 0, respectively
    preds = [1 if pred == "CORRECT" else 0 for pred in preds]
    gts = [1 if gt == "CORRECT" else 0 for gt in gts]

    # Calculate the metrics
    report = classification_report(gts, preds, output_dict=True)

    # Extract the metrics from the classification report
    f1_0 = report['0']['f1-score']
    f1_1 = report['1']['f1-score']

    supp_0 = report['0']['support']
    supp_1 = report['1']['support']

    acc = report['accuracy']
    balanced_acc = balanced_accuracy_score(gts, preds)

    metrics = {
        'f1_0': f1_0,
        'f1_1': f1_1,
        'supp_0': supp_0,
        'supp_1': supp_1,
        'acc': acc,
        'bal_acc': balanced_acc
    }

    return metrics
