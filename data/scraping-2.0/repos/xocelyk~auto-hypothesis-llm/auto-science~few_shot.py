import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt, parse_response
from dotenv import load_dotenv
import os
from config import load_config
from data_loader import split_data
import multiprocessing

config = load_config()
prompts = config['prompts']
# probably should put this in the config file
TIMEOUT = 10


def get_response_with_timeout(prompt, temperature):
    return get_response(prompt, temperature)

def cot_prompt(train_data, test_data, num_shots=16):
    system_content = prompts['SYSTEM_CONTENT_2']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    user_content_3 = prompts['NO_HYPOTHESIS_USER_CONTENT_3']
    messages = [{"role": "system", "content": system_content}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}]
    prompt = create_prompt(num_shots, train_data=train_data, test_data=test_data, messages=messages, train_mode=True, test_mode=True)
    # CHANGED: only reinforce template if we are doing zero shot
    if num_shots == 0:
        prompt.append({"role": "user", "content": user_content_3})
    # user_content_3 = "Is the median house value for houses in this block greater than $200,000? You must answer either '(A) The median house value for houses in this block is greater than $200,000.' OR '(B) The median house value for houses in this block is less than or equal to $200,000.' If you do not follow the answer template someone will die."
    # prompt.append({"role": "user", "content": user_content_3})
    return prompt

def few_shot(train_data, test_validation_data, num_shots=2, verbose=True):
    responses = []
    gts = []
    first = True
    correct = incorrect = invalid = api_timeout = total = 0
    for key in test_validation_data.keys():
        # randomize ICL examples
        train_data_keys = list(train_data.keys())
        np.random.shuffle(train_data_keys)
        train_data_keys = train_data_keys[:num_shots]
        icl_data = {key: train_data[key] for key in train_data_keys}
        test_label = test_validation_data[key]['Label']
        prompt = cot_prompt(icl_data, test_validation_data[key], num_shots=num_shots)
        # Initialize a Pool with one process
        pool = multiprocessing.Pool(processes=1)

        # Call get_response_with_timeout() in that process, and set timeout as 5 seconds
        result = pool.apply_async(get_response_with_timeout, args=(prompt, 0.0))

        try:
            # get the result within 5 seconds
            response_text = result.get(timeout=5)[0]
        except multiprocessing.TimeoutError:
            print("get_response() function took longer than 5 seconds.", end="\r")
            api_timeout += 1
            pool.terminate()  # kill the process
            continue  # go to the next loop iteration
        except openai.error.APIConnectionError:
            print('API Connection Error', end='\r')
            api_timeout += 1
            pool.terminate()  # kill the process
            continue

        pool.close()  # we are not going to use this pool anymore
        pool.join()  # wait for the pool to close by joining
        response = parse_response(response_text)
        responses.append(response)
        gts.append(test_label)
        if response == -1:
            invalid += 1
        elif response == test_label:
            correct += 1
        else:
            incorrect += 1
        total += 1

        # descriptive stats
        tp = np.array([1 if response == 1 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
        fp = np.array([1 if response == 1 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
        tn = np.array([1 if response == 0 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
        fn = np.array([1 if response == 0 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
        recall = tp/(tp + fn)
        precision = tp/(tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)
        verbose = True
        if verbose:
            try: # handle divide by zero
                print('Accuracy:', round(correct/(incorrect + correct), 3), 'Correct:', correct, 'Incorrect:', incorrect,
                      'Invalid:', invalid, 'API Timeout:', api_timeout, 'Total:', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', round(f1, 3), 
                      'Recall:', round(recall, 3), 'Precision:', round(precision, 3), end='\r')
            except ZeroDivisionError:
                print('Accuracy:', 0, 'Correct:', correct, 'Incorrect:', incorrect, 
                      'Invalid:', invalid, 'API Timeout:', api_timeout, 'Total:', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', 0, 
                      'Recall:', 0, 'Precision:', 0, end='\r')
    return {'correct': correct, 'incorrect': incorrect, 'invalid': invalid, 'api_timeout': api_timeout, 'total': total, 
            'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': round(correct/(incorrect + correct), 3)}

def few_shot_one_example(test_icl_data, test_validation_data, num_shots):
    assert 'Label' in test_validation_data.keys(), 'Few Shot One Example takes one example at a time'
    test_icl_data_keys = list(test_icl_data.keys())
    np.random.shuffle(test_icl_data_keys)
    test_icl_data_keys = test_icl_data_keys[:num_shots]
    test_icl_data = {key: test_icl_data[key] for key in test_icl_data_keys}
    test_label = test_validation_data['Label']
    prompt = cot_prompt(test_icl_data, test_validation_data, num_shots=num_shots)
    response_text = get_response(prompt, temperature=0, timeout=TIMEOUT)[0]
    response = parse_response(response_text)
    if response == test_label:
        correct = 1
    else:
        if response == -1:
            correct = -1
        else:
            correct = 0
    return {'response': response, 'correct': correct, 'label': test_label, 'text': response_text}

if __name__ == '__main__':
    config = load_config()
    prompts = config['prompts']
    data_dict = config['data_dict']
    train_data, test_icl_data, test_validation_data = split_data(data_dict, 100, 100, 200)
    test_icl_data_filename = 'data/experiment2/test_icl_data.pkl'
    test_validation_data_filename = 'data/experiment2/test_validation_data.pkl'

    import pickle
    test_icl_data = pickle.load(open(test_icl_data_filename, 'rb'))
    test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))

    test_size = 200
    test_validation_data_keys = list(test_validation_data.keys())
    np.random.shuffle(test_validation_data_keys)
    test_validation_data_keys = test_validation_data_keys[:test_size]
    test_validation_data = {key: test_validation_data[key] for key in test_validation_data_keys}
    results = few_shot(train_data, test_validation_data, num_shots=0, verbose=True)
    print(results)
