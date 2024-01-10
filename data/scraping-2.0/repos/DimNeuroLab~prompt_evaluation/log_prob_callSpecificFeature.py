import signal
import threading
import time
from itertools import product
from random import choice

import numpy as np
import openai
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm

from utils import get_api_key

FEATURES_FILENAME = 'features_new'
ANNOTATION_FILENAME = 'new_majority_annotations'
FEATURES_FILE_PATH = 'data/' + FEATURES_FILENAME + '.tsv'
ANNOTATIONS_FILE_PATH = 'data/' + ANNOTATION_FILENAME + '.tsv'

FEATURES = pd.read_csv(FEATURES_FILE_PATH, sep='\t')
ANNOTATIONS = pd.read_csv(ANNOTATIONS_FILE_PATH, sep='\t')

CURRENT_FEATURE = "1 Goal (1,NaN)"

OPENAI_API_KEY = get_api_key()

DETERMINISTIC_MODEL_NAME = "gpt-3.5-turbo"
PROBABILISTIC_MODEL_NAME = "gpt-3.5-turbo-instruct"

PROMPT_CREATOR_ID = 0

NUMBER_OF_SHOTS = 2
NUMBER_OF_RUNS = 5

IS_DETERMINISTIC_EVALUATION_ENABLED = True
IS_PROBABILISTIC_EVALUATION_ENABLED = True

openai.api_key = OPENAI_API_KEY

YES_STRINGS = {"Yes", "YES", "Y", " Yes", " YES", " Y", "Yes ", "YES ", "Y ", " Yes ", " YES ", " Y "}
NO_STRINGS = {"No", "NO", "N", " No", " NO", " N", "No ", "NO ", "N ", " No ", " NO ", " N"}

Verbose = True
@retry(wait=wait_random_exponential(min=1, max=240), stop=stop_after_attempt(4))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def get_positive_few_shot_example(feature_name, prompt, shots=1):
    try:
        relevant_data = ANNOTATIONS[['prompt', feature_name]]
        filtered_data = relevant_data.loc[(relevant_data['prompt'] != prompt) & (relevant_data[feature_name] == 1)]
        return filtered_data.sample(shots)['prompt'].values
    except:
        return ''


def get_true_label(feature_name, prompt, shots=1):
    try:
        relevant_data = ANNOTATIONS[['prompt', feature_name]]
        filtered_data = relevant_data.loc[relevant_data['prompt'] == prompt]
        return filtered_data.sample(shots)[feature_name].values
    except:
        return ''


def get_negative_few_shot_example(feature_name, prompt, shots=1):
    try:
        relevant_data = ANNOTATIONS[['prompt', feature_name]]
        filtered_data = relevant_data.loc[(relevant_data['prompt'] != prompt) & (relevant_data[feature_name] == 0)]
        return filtered_data.sample(shots)['prompt'].values
    except:
        return ''


class timeoutLinux:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class timeoutWindows:
    def __init__(self, seconds=1, error_message='Timeout '):
        self.seconds = seconds
        self.error_message = error_message + ' ' + str(seconds)

    def handle_timeout(self):
        print("timemout")
        raise TimeoutError(self.error_message)

    def __enter__(self):
        # signal.signal(signal.SIGALRM, self.handle_timeout)
        # signal.alarm(self.seconds)
        # print("seconds", self.seconds)
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.start()

    def __exit__(self, type, value, traceback):
        self.timer.cancel()
        # signal.alarm(0)


def create_prompt_zero(eval_prompt, feature, shots):
    # collect necessary data
    feature_data = FEATURES.loc[FEATURES['feature_name'] == feature]
    feature_description = feature_data['prompt_command'].iloc[0]

    # prepare prompt examples
    positive_few_shot = get_positive_few_shot_example(feature, eval_prompt, shots=shots)
    negative_few_shot = get_negative_few_shot_example(feature, eval_prompt, shots=shots)

    # join all prompts into a single string
    formatted_positive_few_shot = '\n'.join(positive_few_shot)
    formatted_negative_few_shot = '\n'.join(negative_few_shot)

    # define eval_string structure
    eval_string = f"""Me: Check if this feature:
            {feature_description}\n
            is present in the following prompts, answer with YES or NO\n
            {formatted_positive_few_shot}\n
            You: Yes\n
            Me: and in the following prompt?
            {formatted_negative_few_shot}\n
            You: No\n

            Me: and in the following prompt?
            {eval_prompt}\n
            You: 
            """
    return eval_string, feature_description


def create_prompt(eval_prompt, feature, shots):
    # Extract feature description
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature, 'prompt_command'].iloc[0]

    # Get positive and negative few shot examples
    pos_few_shot1, neg_few_shot1 = map(get_few_shot_examples,
                                       [feature] * 2,
                                       [eval_prompt] * 2,
                                       [shots] * 2,
                                       [True, False])

    pos_few_shot2, neg_few_shot2 = map(get_few_shot_examples,
                                       [feature] * 2,
                                       [eval_prompt] * 2,
                                       [shots] * 2,
                                       [True, False])

    # Create the evaluation string
    eval_string = f"""Me: Check if this feature:
            {feature_description}\n
            is present in the following prompts, answer with YES or NO\n
            {pos_few_shot1}\n
            You: Yes\n
            Me: and in the following prompt?
            {neg_few_shot1}\n
            You: No\n

            Me: and in the following prompt?
            {neg_few_shot2}\n
            You: No\n
            Me: and in the following prompt?
            {pos_few_shot2}\n
            You: Yes\n

            Me: and in the following prompt?
            {eval_prompt}\n
            You: \n
            """
    return eval_string, feature_description


def get_few_shot_examples(feature, eval_prompt, shots, is_positive):
    # Select function based on is_positive flag
    func = get_positive_few_shot_example if is_positive else get_negative_few_shot_example
    # Call the function and join the examples
    examples = '\n'.join(func(feature, eval_prompt, shots=shots))

    return examples


def create_prompt_inverted(eval_prompt, feature, shots):
    """
    Generate an evaluation prompt string following a specific pattern.
    The prompt command related to the feature is fetched from a global 'FEATURES' DataFrame.

    Args:
        eval_prompt (str): A string containing the evaluation prompt.
        feature (str): A string representing the feature.
        shots (int): The number of shots.

    Returns:
        Tuple[str, str]: Returns a tuple with evaluation string and feature description.
    """
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature, 'prompt_command'].iloc[0]

    negative_shots = lambda: '\n'.join(get_negative_few_shot_example(feature, eval_prompt, shots=shots))
    positive_shots = lambda: '\n'.join(get_positive_few_shot_example(feature, eval_prompt, shots=shots))

    eval_string = f"""Me: Answer with Yes or No if this feature:
            {feature_description}

            is present in the following prompt:

            {negative_shots()}

            You: No
            Me: and in the following prompt?

            {positive_shots()}

            You: Yes

            Me: and in the following prompt?

            {positive_shots()}

            You: Yes
            Me: and in the following prompt?

            {negative_shots()}

            You: No

            Me: and in the following prompt?

            {eval_prompt}

            You: 
            """
    return eval_string, feature_description


def create_prompt_random_2(eval_prompt, feature, shots):
    """
    Given a feature, an evaluation prompt and a number of shots, it formats a string
    prompt that asks whether the feature is present in different prompts.

    Parameters:
    eval_prompt (str): The evaluation prompt.
    feature (str): The specific feature to be evaluated in the prompts.
    shots (int): The number of prompts to be used for the evaluation.

    Returns:
    tuple: The formatted string prompt that asks whether the feature is present
           and the description of the feature.
    """
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature, 'prompt_command'].iloc[0]
    prompt_intro = (f"Me: Answer with Yes or No if this feature:\n{feature_description}\n"
                    "applies ...\nto the following prompt:\n")

    random_prompts = []
    for _ in range(shots):
        positive = '\n'.join(get_positive_few_shot_example(feature, eval_prompt, shots=1))
        negative = '\n'.join(get_negative_few_shot_example(feature, eval_prompt, shots=1))
        order = np.random.choice(2)

        if order == 1:
            shots_str = (f"{negative}\nYou: No\n Me: to the following prompt:\n"
                         f"{positive}\nYou: Yes\nMe: to the following prompt:\n")
            random_prompts.append(shots_str)
        else:
            shots_str = (f"{positive}\nYou: Yes\nMe: to the following prompt:\n"
                         f"{negative}\nYou: No\nMe: to the following prompt:\n")
            random_prompts.append(shots_str)

    final_prompt = (f"{prompt_intro}{''.join(random_prompts)}\n{eval_prompt}\nYou:\n")
    return final_prompt, feature_description


def create_prompt_random_3(eval_prompt, feature, shots):
    '''
    Generates a string to ask a series of questions to a user. The questions are based on a given feature and
    a list of positive and negative examples are generated for each shot. The order of appearance of positive
    or negative examples is random.

    :param eval_prompt: prompt for evaluation
    :param feature: feature based on which questions are formed
    :param shots: number of shots
    :return: a tuple of generated string and feature description
    '''
    # get feature description from FEATURES dataframe
    feature_description = FEATURES[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    # init empty lists for few shot examples
    positive_few_shot, negative_few_shot = [], []

    # init list for storing results
    results = [f"Me: Answer with Yes or No if this description:\n {feature_description}\n",
               "applies to the following prompt:\n"]

    for i in range(shots):
        # generate positive and negative few shot examples
        positive_few_shot.append('\n'.join(get_positive_few_shot_example(feature, eval_prompt, shots=1)))
        negative_few_shot.append('\n'.join(get_negative_few_shot_example(feature, eval_prompt, shots=1)))

        # randomly determine the order of positive and negative examples
        if choice([True, False]):
            results.extend([f"{negative_few_shot[i]}\n",
                            "You: No\n",
                            "Me: and in the following prompt?\n",
                            f"{positive_few_shot[i]}\n",
                            "You: Yes\n",
                            "Me: and in the following prompt?\n"])
        else:
            results.extend([f"{positive_few_shot[i]}\n",
                            "You: Yes\n",
                            "Me: and in the following prompt?\n",
                            f"{negative_few_shot[i]}\n",
                            "You: No\n",
                            "Me: and in the following prompt?\n"])

    results.extend([f"{eval_prompt}\n", "You: \n"])

    # join results list into single string with newlines
    results_string = '\n'.join(results)

    return results_string, feature_description


def create_prompt_random(eval_prompt, feature, shots):
    """
    Creates a prompt for the user to answer if a feature is present in it.
    Args:
        eval_prompt (str): The evaluation prompt.
        feature (str): The feature under consideration.
        shots (int): The number of positive/negative examples.

    Returns:
        str: The prepared prompt.
        str: The feature description.
    """

    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature, 'prompt_command'].values[0]

    eval_string = (
        f"Me: Answer with Yes or No if this feature:\n"
        f"{feature_description}\n"
        "is present in the following prompt:\n"
    )

    for _ in range(shots):
        positive_example = "\n".join(get_positive_few_shot_example(feature, eval_prompt, shots=1))
        negative_example = "\n".join(get_negative_few_shot_example(feature, eval_prompt, shots=1))

        if np.random.choice(2, 1) == 1:
            eval_string += (
                f"{negative_example}\n"
                f"You: No\n"
                f"Me: and in the following prompt?\n"
                f"{positive_example}\n"
                f"You: Yes\n"
                f"Me: and in the following prompt?\n"
            )
        else:
            eval_string += (
                f"{positive_example}\n"
                f"You: Yes\n"
                f"Me: and in the following prompt?\n"
                f"{negative_example}\n"
                f"You: No\n"
                f"Me: and in the following prompt?\n"
            )

    eval_string += (
        f"{eval_prompt}\n"
        f"You: \n"
    )

    return eval_string, feature_description


def create_prompt_revised(eval_prompt, feature, num_prompts):
    """Create a prompt evaluation string.
    Args:
    eval_prompt (str): an evaluation prompt.
    feature (str): a given feature.
    num_prompts (int): quantity of prompts.
    Returns:
    str: an evaluation string.
    str: feature description.
    """
    feature_description = FEATURES.loc[FEATURES['feature_name'] == feature]['prompt_command'].iloc[0]

    # Generate few shot examples for each type (positive & negative) and quantity of prompts
    # Each generated list of strings are then joined as a new line separated string
    few_shot_examples = {
        "positive": [],
        "negative": []
    }
    for each_type in few_shot_examples.keys():
        for _ in range(num_prompts):
            # Select correct function based on the type
            few_shot_func = get_positive_few_shot_example if each_type == "positive" else get_negative_few_shot_example
            # Execute the function
            few_shot_example = few_shot_func(feature, eval_prompt, shots=NUMBER_OF_SHOTS)
            # Append joined examples to the list
            few_shot_examples[each_type].append('\n'.join(few_shot_example))

    # Format the output string
    eval_string = f"""Me: Answer with Yes or No if this feature:
        {feature_description}\n
        is present in the following prompt:\n
        {few_shot_examples["positive"][0]}\n
        You: Yes\n
        Me: and in the following prompt?\n
        {few_shot_examples["negative"][0]}\n
        You: No\n
        Me: and in the following prompt?\n
        {few_shot_examples["negative"][1]}\n
        You: No\n
        Me: and in the following prompt?\n
        {few_shot_examples["positive"][1]}\n
        You: Yes\n

        Me: and in the following prompt?\n
        {eval_prompt}\n
        You: \n
        """
    return eval_string, feature_description


def create_prompt_gregor(eval_prompt, feature, shots):
    feature_info = FEATURES.loc[FEATURES['feature_name'] == feature]
    feature_description = feature_info['prompt_command'].iloc[0]

    positive_examples = prepare_shot_examples(get_positive_few_shot_example, feature, eval_prompt, shots)
    negative_examples = prepare_shot_examples(get_negative_few_shot_example, feature, eval_prompt, shots)

    eval_string = create_eval_string(
        feature_description,
        positive_examples,
        negative_examples,
        eval_prompt
    )

    return eval_string, feature_description


def prepare_shot_examples(get_examples_function, feature, eval_prompt, shots):
    shot_examples = get_examples_function(feature, eval_prompt, shots=shots)
    return format_shot_examples(shot_examples)


def format_shot_examples(shot_examples):
    enumerated_examples = [f'Prompt {idx + 1} : {val}'
                           for idx, val in enumerate(shot_examples)]
    return '\n'.join(enumerated_examples)


def create_eval_string(feature_desc, pos_examples, neg_examples, prompt):
    return (f"Given the following feature:\n"
            f"{feature_desc}\n"
            "\nThe feature is present in the following prompts:\n"
            f"{pos_examples}\n"
            "\nThe feature is not present in the following prompts:\n"
            f"{neg_examples}\n"
            "\nTell me whether the feature is present in the prompt given "
            "below. Formalize your output as a json object, where the key "
            "is the feature description and the value is 1 if the feature "
            "is present or 0 if not.\n\nPrompt:\n"
            f"{prompt}")


def evaluate_prompt_both(feature_list, eval_prompt, shots=1, prompt_creator=2):
    creator_dict = {
        0: create_prompt_zero,
        1: create_prompt,
        2: create_prompt_inverted,
        3: create_prompt_revised,
        4: create_prompt_random,
        5: create_prompt_random_2,
        6: create_prompt_random_3
    }

    for feature in feature_list:
        if prompt_creator in creator_dict:
            eval_string, feature_description = creator_dict[prompt_creator](eval_prompt, feature, shots)

        conversation = [{'role': 'system', 'content': eval_string}]

        eval_string_prob = eval_string
        prompt_annotations_prob = None
        prompt_annotations_det = None

        if Verbose:
            print('*'*60)
            print("eval_string_prob")
            print(eval_string_prob)

        if IS_PROBABILISTIC_EVALUATION_ENABLED:
            prompt_annotations_prob = evaluate_prompt_logits(
                feature=feature,
                eval_string=eval_string_prob,
                feature_description=feature_description,
                eval_prompt=eval_prompt
            )
        if IS_DETERMINISTIC_EVALUATION_ENABLED:
            prompt_annotations_det = evaluate_prompt_det(
                feature=feature,
                feature_desc=feature_description,
                conversation=conversation,
                evaluation_prompt=eval_prompt
            )

        return prompt_annotations_prob, prompt_annotations_det


def evaluate_prompt_logits(feature, eval_string, feature_description, eval_prompt):
    global not_good_response
    prompt_annotations = [eval_prompt]

    response = None

    for _ in range(5):
        try:
            time.sleep(0.5)
            with timeoutWindows(seconds=100):
                response = completion_with_backoff(
                    model=PROBABILISTIC_MODEL_NAME,
                    prompt=eval_string,
                    max_tokens=1,
                    temperature=0,
                    logprobs=2,
                    logit_bias={},
                )
            if response['choices'][0]["logprobs"]["tokens"][0] in YES_STRINGS | NO_STRINGS:
                break
            else:
                print(f"{80 * '+'}\nbad response\n{response}\n{80 * '+'}")
        except Exception as E:
            print("exx")
            print(E)
            print('Timeout, retrying...')

    gt = get_true_label(feature, eval_prompt)

    if response and response['choices'][0]["logprobs"]["tokens"][0] in YES_STRINGS | NO_STRINGS:

        value = response['choices'][0]["logprobs"]["token_logprobs"][0]
        response_value_Y = value if response['choices'][0]["logprobs"]["tokens"][0] in YES_STRINGS else -100
        response_value_N = value if response['choices'][0]["logprobs"]["tokens"][0] in NO_STRINGS else -100

        print(f"Y: {response_value_Y}; F: {response_value_N}; GT: {gt}")
        if Verbose:
            print('*' * 60)
            print("response_prob")
            print(response_value_Y,response_value_N)
            print("Gt")
            print(gt)
            print("response_prob string")
            print(response)


    else:
        response_value_Y = response_value_N = -100
        not_good_response += 1
        if Verbose:
            print('*' * 60)
            print("response_prob")
            print("bad response")
            print(response)
            print("Gt")
            print(gt)

    # print(response_value_Y, response_value_N)
    prompt_annotations.extend([response_value_Y, response_value_N, gt[0]])

    return prompt_annotations


def evaluate_prompt_det(feature, feature_desc, conversation, evaluation_prompt):
    """
    Evaluate the given prompt and return the annotations.

    Args:
        feature (str): The feature to evaluate.
        feature_desc (str): The feature's description.
        conversation (list): The user's conversation.
        evaluation_prompt (dict): The evaluation prompt.
        debug (bool, optional): Whether to print debug info. Defaults to True.

    Returns:
        list: The annotations from the evaluation.
    """
    feature_list = FEATURES['feature_name'].tolist()
    prompt_annotations = [evaluation_prompt]
    response = None
    try:
        response = openai.ChatCompletion.create(model=DETERMINISTIC_MODEL_NAME, messages=conversation)
        # print("DETERMINISTIC RESPONSE")
        response_parsed = response['choices'][0]['message']['content']
    except:
        print("DETERMINISTIC RESPONSE")
        response_parsed = {feature_desc: -1}

    gt = get_true_label(feature, evaluation_prompt)

    if response:
        response_key = next(iter(response), None)

        response_value = response_parsed.get(response_key, -1) if isinstance(response_parsed, dict) else -1
        prompt_annotations.append(response_value)
        if Verbose:
            print('*' * 60)
            print("response_det")
            print(response_value)
            print("Gt")
            print(gt)
            print("response_prob string")
            print(response)
    else:
        prompt_annotations.append(-1)

    prompt_annotations.append(gt[0])

    return prompt_annotations


def build_column_names(annotations):
    df_column_names_1 = ["Y", "N"]
    df_column_names = [list(annotations.columns)[0]]
    df_column_names.extend(df_column_names_1)
    df_column_names.insert(0, "feature_name")

    # Add classification
    det_column_names = df_column_names[:-2]
    det_column_names.append("class")

    # Add ground truth
    df_column_names.append("gt")
    det_column_names.append("gt")

    return df_column_names, det_column_names


def collect_data_values(features, annotations, num_shots):
    df_values = []
    det_annotations_data = []
    #for _ in range(num_shots):
    prompts = annotations["prompt"].tolist()

    for prompt in tqdm(prompts):
        for feature_name in features["feature_name"]:
            prompt_annotations, det_annotations = evaluate_prompt_both([feature_name], prompt, shots=num_shots)

            prompt_annotations.insert(0, feature_name)
            det_annotations.insert(0, feature_name)

            df_values.append(prompt_annotations)
            det_annotations_data.append(det_annotations)
        break

    return df_values, det_annotations_data


def save_to_file(data_values, column_names, model_name, deterministic_model_name, num_shots, creator_id,
                 features_filename, annotation_filename, suffix):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"output/{model_name} {deterministic_model_name}_evaluation_log_shots_{num_shots}promptgen_{creator_id}_features_file_{features_filename}_annotation_file_{annotation_filename}_{timestr}nobias_{suffix}.tsv"

    print(data_values)
    print(column_names)

    result_data = pd.DataFrame(np.array(data_values), columns=column_names)
    result_data.to_csv(filename, sep="\t", index=False)


def main(num_runs):
    df_column_names, det_column_names = build_column_names(ANNOTATIONS)

    print(df_column_names)
    print(det_column_names)

    for _ in range(num_runs):
        df_values, det_annotations_data = collect_data_values(FEATURES, ANNOTATIONS, NUMBER_OF_SHOTS)

        save_to_file(
            df_values, df_column_names, PROBABILISTIC_MODEL_NAME, DETERMINISTIC_MODEL_NAME, NUMBER_OF_SHOTS,
            PROMPT_CREATOR_ID, FEATURES_FILENAME, ANNOTATION_FILENAME, "nobias"
        )
        print(det_annotations_data)
        save_to_file(
            det_annotations_data, det_column_names, PROBABILISTIC_MODEL_NAME, DETERMINISTIC_MODEL_NAME,
            NUMBER_OF_SHOTS, PROMPT_CREATOR_ID, FEATURES_FILENAME, ANNOTATION_FILENAME, "nobias_DET"
        )


if __name__ == "__main__":
    main(NUMBER_OF_RUNS)
