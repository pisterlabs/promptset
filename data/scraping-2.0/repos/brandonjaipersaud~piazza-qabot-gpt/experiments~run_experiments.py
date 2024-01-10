"""
Main script which takes a CSV and does classification, answering and saving of model answer logprobs
"""

from io import TextIOWrapper
import sys
import re
import os
import json
import copy
from pathlib import Path
from datetime import datetime

import openai
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from openai_interface.prompt_gpt3 import prompt_gpt3


from experiments.hyperparameters import *
from utils.utils import is_empty


# save current experiment in a new directory. Use timestamp to create a unique directory for each experiment.
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = f'{general_params["run_dir"]}{timestamp}_{general_params["run_suffix_tag"]}'



def naive_baseline():
    """Pick most common class in csv?
    """
    #TODO
    pass



def answer_question(row, store_logprobs=None, response_file:TextIOWrapper=None):

    """Answer a single question in the CSV file. Helper function used by inference().
        Precondition: row contains field question 
        Creates a new field called model answer and puts the classification there
    """
    question = row['question']
    prompt=model_params['prompt'].format(question)

    response = prompt_gpt3(prompt)
    response_file.write(str(response))

    text = response["choices"][0]["text"]

    row['model_answer'] = text


    # save logprobs
    if store_logprobs:
        logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
        row['logprobs'] = logprobs
        # print(row)
   

    if general_params['verbose']:
        print(f"Question: {question}\n")
        print(f"Answer: {text}\n")
        print('-------------------------------')
    return row



def classify_question(row, store_logprobs=None, response_file:TextIOWrapper=None):
    """Classify a single question in the CSV file. Helper function used by inference().
        Precondition: row contains fields question and target
        Creates a new field called predicted category and puts the classification there

        Targets may/may not be avaliable at the time of classification
    """
    
    question = row['question']
    prompt=model_params['prompt'].format(question)

  
    response = prompt_gpt3(prompt)
    response_file.write(str(response))

    if response:
        text = response["choices"][0]["text"]

   
    row['classification'] = text

    # save logprobs
    #print(response)
    if store_logprobs and response:
        logprobs = response["choices"][0]['logprobs']['token_logprobs']
        row['logprobs'] = logprobs


    if general_params['verbose']:
        print(f"Question: {question}\n")
        print(f"Classification: {text}\n")
        print('-------------------------------')
    return row


def load_dataset():

    if type(dataset_params['dataset_path']) == str:
        return pandas.read_csv(dataset_params['dataset_path'])
    elif type(dataset_params['dataset_path']) == pandas.DataFrame:
        return dataset_params['dataset_path']
    else:
        raise AssertionError


def inference():
    """ Takes a CSV of questions, adds a new column: "classification" and fills it with model predictions.
        Precondition: CSV contains columns: "question" and "target"

        Potential Changes:
            - operate on df as well as files
            - prompt specify directly as well as in file

        Notes
            - ensure original csv is not overwritten
    """
    # log openai api response objects
    response_path = RUN_DIR + '/' + 'openai_response.csv'

    df = load_dataset()
    

    if dataset_params['shuffle']:
        df = df.sample(frac=1)

    # setup new cols to be filled out
    if control_params["classify"]:
        df['classification'] = ""
    else:
        df['model_answer'] = ""

    if model_params['logprobs']:
        df['logprobs'] = ""

    if (dataset_params['num_samples']):
        df = df.head(dataset_params['num_samples'])

    if general_params['verbose']:
        print(f'Using prompt:\n\n{model_params["prompt"]}\n\n')
        print("-" * 50)


    if control_params['classify']:
        with open(RUN_DIR + '/' + 'openai_response_classify.txt', 'w') as f:
            print(f'Saving openai api answer response to: {RUN_DIR + "/" + "openai_response_classify.txt"}')
            df = df.apply(lambda row: classify_question(row, store_logprobs=model_params['logprobs'], response_file=f), axis=1)
            out_path = RUN_DIR + '/' + 'classify_inference.csv'
            print(f'Saving Classify CSV to: {out_path}')
    else:
        with open(RUN_DIR + '/' + 'openai_response_answer.txt', 'w') as f:
            print(f'Saving openai api answer response to: {RUN_DIR + "/" + "openai_response_answer.txt"}')
            df = df.apply(lambda row: answer_question(row, store_logprobs=model_params['logprobs'], response_file=f), axis=1)
            out_path = RUN_DIR + '/' + 'answer_inference.csv'
            print(f'Saving Answer CSV to: {out_path}')
   
    df.to_csv(out_path, index=False)



def evaluate():
    """Performs evaluation on the csv file in the run_dir. Saves metrics to a new file: eval.txt in the run_dir.

    Metrics Tracked:
        - accuracy
        - f1 micro + macro
        - confusion matrix

    Adds special <unknown-category> token for model predictions that are not in the classification categories in the dataframe.
    - i.e. it's possible for gpt3 to output something other than the labels you tell it to output (i.e. answerable vs. other)

    Precondition:
        1. data csv has columns named: target and classification

    """
    inference_path = RUN_DIR + '/' + 'classify_inference.csv'

    df = pandas.read_csv(inference_path)

    targets = df['target'].str.lower().str.strip()
   
    predicted_categories = df['classification'].str.lower().str.strip()
    predicted_categories[~predicted_categories.isin(dataset_params['labels'])] = '<unknown-category>'
    dataset_params['labels'].append("<unknown-category>")

    # compute and save cm if specified
    cm = confusion_matrix(targets, predicted_categories, labels=dataset_params['labels'])
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset_params['labels'])
    cmd.plot()

    if (general_params['save_run']):
        save_dir =  RUN_DIR + '/' + 'confusion_matrix.png'
        print(f'Saving to: {save_dir}')
        fig = plt.gcf()
        fig.set_size_inches(15, 15)

        fig.savefig(save_dir)

    # compute and save eval metrics if specified

    f1_macro = f1_score(targets, predicted_categories, labels=dataset_params['labels'], average='macro')
    f1_micro = f1_score(targets, predicted_categories, labels=dataset_params['labels'], average='micro')
    accuracy = accuracy_score(targets, predicted_categories)
    num_correct = accuracy_score(targets, predicted_categories, normalize=False)
    total = len(targets)

    print(f'accuracy: {accuracy} ({num_correct}/{total})')
    #print(f'f1_macro: {f1_macro}')
    #print(f'f1_micro: {f1_micro}')

    if (general_params['save_run']):
        eval_path = RUN_DIR + '/' + 'eval.txt'

        with open(eval_path, 'w') as f:
            f.write(f'Accuracy: {accuracy} ({num_correct}/{total})\n')
            f.write(f'f1_macro: {f1_macro}\n')
            f.write(f'f1_micro: {f1_micro}\n')

    # rename directory with val score?





def plot_metrics():
    """ Plots metrics across different runs. Use this for plotting prompt vs. eval score across different runs.

    Iterates through runs specified in control_params['plot_dirs'] and extracts their eval metrics stored in
    eval.txt.

    Precondition: eval.txt has the format:
        Accuracy: <score>
        f1_macro: <score>
        f1_micro: <score>
    """

    run_dirs = control_params['plot_dirs']
    accuracy, f1_macro, f1_micro, labels = [], [], [], []
    scores = [accuracy, f1_macro, f1_micro]

    for timestamp in run_dirs:
        for run in os.listdir(general_params['run_dir']):
            if timestamp in run:
                run_tag = None
                eval_file = general_params['run_dir']  + run + '/' + 'eval.txt'
                print(f'opening eval file: {eval_file}')
                with open(eval_file, 'r') as f:
                    for idx, line in enumerate(f):
                        scores[idx].append(float(line.replace('\n', '').split(' ')[-1]))
                        param_file = general_params['run_dir']  + run + '/' + 'hyperparameters.json'
                        print(f'opening hyperparameters file: {param_file}')
                        with open(param_file) as f2:
                         params = json.load(f2)
                         run_tag = params['general_params']['run_suffix_tag']
                labels.append(run_tag)


    # plot metrics

    plt.plot(labels, scores[0], label='accuracy', marker='o')
    plt.plot(labels, f1_macro, label='f1_macro', marker='o')
    plt.plot(labels, f1_micro, label='f1_micro', marker='o')
    plt.legend()
    plt.title('Prompt vs. Evaluation Score')
    plt.xlabel('Prompt')
    plt.ylabel('Evaluation Score')
    plt.xticks(rotation=15)

    fig = plt.gcf()
    fig.set_size_inches(12, 12)


    save_plot_path = RUN_DIR + '/' + 'plot.png'
    fig.savefig(save_plot_path)




def write_dict_to_file(d:dict, file):
    """Nicely format a dict in a text file
    """

    #print(d)
    for k,v in d.items():
        #print(k, v)
        file.write(f'    {k} : {v}\n')


def get_instr_answer_logprobs(response, print_answer=False, debug=False):
    """Given an openai response object, extract instructor answer logprobs from it"""
    text = response['choices'][0]['text']
    text_offset = response['choices'][0]['logprobs']['text_offset']
    token_logprobs = response['choices'][0]['logprobs']['token_logprobs']

    # add 2 to get start of instructor answer: Answer:<instr answer>
    answer_index = text.find('Answer:')
    # token = Answer
    answer_token_index = text_offset.index(answer_index)
    # token = Answer: <instr-answer>
    answer_logprobs = token_logprobs[answer_token_index + 2:]

    if print_answer:
        print(text[answer_index:])

    tokens = response['choices'][0]['logprobs']['tokens']
    if tokens[answer_token_index].lower() != 'answer' or tokens[answer_token_index+1].lower() != ':':
        print('Token index of Answer or : does not match up!')

    if debug:
        print(text[answer_index:])
        print(text)
        print(answer_token_index)
        print(answer_logprobs)
        return text, answer_token_index, answer_logprobs
    return answer_logprobs


def generate_logprobs(row, response_file:TextIOWrapper=None, specify_question=None, specify_answer=None):
    if specify_question:
        question = specify_question
        answer = specify_answer
    else:
        question = row['question']
        answer = row['instructor_answer']

    if not is_empty(answer):
        prompt=model_params['prompt'].format(question, answer)
        print(f'Question: {question}')
        print(f'Instructor Answer: {answer}')
        print('-' * 40)

        response = prompt_gpt3(prompt)
       
        if response_file:
            response_file.write(f'Question: {question}')
            response_file.write(f'Instructor Answer: {answer}')
            response_file.write(str(response))

         # for debugging
        if specify_question:
            print('DEBUGGING')
            logprobs = get_instr_answer_logprobs(response, debug=True)
        else:
            logprobs = get_instr_answer_logprobs(response)

        # for debugging
        if specify_question:
            print(response)
            return logprobs
        
        row['instructor_answer_logprobs'] = logprobs

   
    return row



def generate_instructor_answer_logprobs():
    """
    Task Description:
    Question
    Answer: <instructor answer>
    generate logprobs
    """
    df:pandas.DataFrame = load_dataset()
    if (dataset_params['num_samples']):
        df = df.head(dataset_params['num_samples'])
    df["instructor_answer_logprobs"] = ""

    with open(RUN_DIR + '/' + 'openai_response_instr_answer_logprobs.txt', 'a') as f:
            print(f'Saving openai api answer logprobs response to: {RUN_DIR + "/" + "openai_response_instr_answer_logprobs.txt"}')
            df = df.apply(lambda row: generate_logprobs(row, response_file=f), axis=1)
            out_path = RUN_DIR + '/' + 'instr_answer_logprobs.csv'
            print(f'Saving instr_answer_logprobs csv to: {out_path}')
    
    df.to_csv(out_path, index=False)

    # save and return df
    return df




def log_run_params():
    """log parameters associated with experiment to hyperparameters.txt and hyperparameters.json
    This makes it easy to see how hyperparams affect performance.
    """
    param_filename = RUN_DIR + '/' + 'hyperparameters'

    if (general_params['save_run']):
        print(f'Saving hyperparameters to: {param_filename}')
        save_dataset = None
    

        if type(dataset_params['dataset_path']) == pandas.DataFrame:
            print("Saving dataset")
            save_dataset = dataset_params['dataset_path']
            dataset_params['dataset_path'] = 'DataFrame'
        
        params = {
            "general_params" : general_params,
            "dataset_params" : dataset_params,
            "control_params" : control_params,
            "model_params" : model_params,
        }

        

        with open(param_filename + '.json', 'w') as f:
            json.dump(params, f)


        with open(param_filename + '.txt', 'w') as f:
            f.write(f'General Params:\n\n')
            write_dict_to_file(general_params, f)

            f.write(f'\n\nDataset Params:\n\n')
            write_dict_to_file(dataset_params, f)

            f.write(f'\n\nControl Params:\n\n')
            write_dict_to_file(control_params, f)

            f.write(f'\n\nModel Params:\n\n')
            write_dict_to_file(model_params, f)


    if (general_params['verbose']):
        print(f'General Params:\n')
        print(general_params)

        print(f'\n\nDataset Params:\n')
        print(dataset_params)

        print(f'\n\nControl Params:\n')
        print(control_params)


        print(f'\n\nModel Params:\n')
        print(model_params)




    if type(save_dataset) == pandas.DataFrame:
        print("Restoring dataset")
        dataset_params['dataset_path'] = save_dataset




def main():

    if not is_empty(general_params['openai-key-path']):
        openai.api_key_path = general_params['openai-key-path']

    # create run dir if not exists
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

    # 3 control options for now
    if control_params["generate_instructor_answer_logprobs"]:
        print(f'Generating logprobs on the instructor answer specified in: {dataset_params["dataset_path"]}')
    
    else:

        if control_params['classify']:
            print(f'Performing classification on {dataset_params["dataset_path"]}')
        else:
            print(f'Performing answering on {dataset_params["dataset_path"]}')


    if (general_params['save_run']):
        print(f'Run will be saved in: {RUN_DIR}')
        print(general_params['run_dir'])

    log_run_params()

    if control_params["generate_instructor_answer_logprobs"]:
        generate_instructor_answer_logprobs()

    else:

        if (control_params['inference']):
            print(f'Performing inference: \n\n')
            inference()


        if (control_params['evaluate']):
            print(f'Performing evaluation: \n\n')
            evaluate()


    if (control_params['plot_metrics']):
        print(f'Plotting metrics: \n\n')
        plot_metrics()




if __name__ == "__main__":
    main()




"""
TODO: naive baseline -- choose common class

inference, eval go together while plot is separate

might want to plot across diff runs


other baselines:
 - naive -- pick the most common class
 - rnn, lstm, gru, etc.


 accuracy overlaps with f1_micro
"""
