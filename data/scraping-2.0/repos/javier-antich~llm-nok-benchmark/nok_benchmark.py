from dotenv import load_dotenv
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import pandas as pd
import math
import time
import datetime
import os
import json
from calendar import timegm
from datetime import datetime,timedelta
import yaml
import openai
import requests
import ast
import csv
import matplotlib.pyplot as plt
import re
import io
import sys
import seaborn as sns
import argparse
from tabulate import tabulate
from joblib import Parallel,delayed
from threading import Thread 


load_dotenv()  # take environment variables from .env.

if os.environ.get('DEBUG')=="True":
    DEBUG_MODE = True
else:
    DEBUG_MODE = False

OPEN_ROUTER_KEY = os.environ.get('OPENROUTER_KEY')
OPENROUTER_REFERRER = "https://github.com/javier-antich/llm-nok-benchmark"
OPENROUTER = True
OPEN_AI_KEY = os.environ.get('OPENAI_KEY')
BASELINE_MODEL = 'openai/gpt-4-1106-preview'


def load_excel_to_dataframe(file_name):
    """
    Loads an Excel file into a pandas DataFrame.

    :param file_name: The path to the Excel file.
    :return: A pandas DataFrame containing the data from the Excel file.
    """
    try:
        df = pd.read_excel(file_name)
        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None


def load_csv_with_columns(file_name, table_columns):
    # Read the CSV file without headers
    df = pd.read_csv(file_name, header=None)
    
    # Assign column names from the provided list
    df.columns = table_columns
    
    return df


def read_single_column_csv(file_name):
    result = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming there's only one column per row
            if row:  # Check if the row is not empty
                result.append(row[0])
    return result


def extract_json(s):
    pattern1 = r'```json(.*?)```'
    pattern2 = r'```(.*?)```'
    pattern3 = r'{.*?}'
    matches = re.findall(pattern1, s, re.DOTALL)
    if DEBUG_MODE:
        print('Fiding json object pattern1')
        print(matches)
    if len(matches)>0:
        return matches[0]
    else:
        matches = re.findall(pattern2, s, re.DOTALL)
        if DEBUG_MODE:
            print('Fiding json object pattern2')
            print(matches)
        if len(matches)>0:
            return matches[0]
        else:
            matches = re.findall(pattern3, s, re.DOTALL)
            if DEBUG_MODE:
                print('Fiding json object pattern3')
                print(matches)
            if len(matches)>0: return matches[0]
            else: return s


def get_completion(prompt, model="gpt-3.5-turbo",DEBUG_MODE=False,use_openai=False):
    
    openai.api_key = OPEN_ROUTER_KEY
    openai.api_base = "https://openrouter.ai/api/v1"
        
    if use_openai:
        openai.api_base = "https://api.openai.com/v1"
        openai.api_key = OPEN_AI_KEY
        model = model.split('/')[1]

    if DEBUG_MODE: print('Prompt: ',prompt)
    messages = [{"role": "user", "content": prompt}]
    request_time = datetime.now().timestamp()
    try:
        response = openai.ChatCompletion.create(
            model=model,
            headers={
            "HTTP-Referer": OPENROUTER_REFERRER
            },
            messages=messages,
            max_tokens=2000,
            temperature=0, # this is the degree of randomness of the model's output
        )
        response_time = datetime.now().timestamp()
        if DEBUG_MODE: print(response)

    except openai.error.Timeout as e:
    #Handle timeout error, e.g. retry or log
        print(f"OpenAI API request timed out: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass
    except openai.error.APIError as e:
    #Handle API error, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass
    except openai.error.APIConnectionError as e:
    #Handle connection error, e.g. check network or log
        print(f"OpenAI API request failed to connect: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass
    except openai.error.InvalidRequestError as e:
    #Handle invalid request error, e.g. validate parameters or log
        print(f"OpenAI API request was invalid: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass
    except openai.error.AuthenticationError as e:
    #Handle authentication error, e.g. check credentials or log
        print(f"OpenAI API request was not authorized: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        print(response)
        pass
    except openai.error.PermissionError as e:
    #Handle permission error, e.g. check scope or log
        print(f"OpenAI API request was not permitted: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass
    except openai.error.RateLimitError as e:
    #Handle rate limit error, e.g. wait or log
        print(f"OpenAI API request exceeded rate limit: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        response = {
             'choices':[{'message':{'content':'{"error":"'+str(e)+'"}'}}]
        }
        pass
    try:
        output_dict = json.loads(response['choices'][0]['message']["content"])
        return response['choices'][0]['message']["content"]
    except:
        if 'choices' in response:
            return extract_json(response['choices'][0]['message']["content"])
        else:
            print(f"An unexpected error occurred")
            response = {
             'choices':[{'message':{'content':'{"error":"no choices in response"}'}}]
            }
            return response['choices'][0]['message']["content"]



#Prompt templates for the benchmark

command_request_prompt_template = """
    A request for information from a certain network device is presented in the field TARGET
    The network device operating system is provided in the field VENDOR_OS
    REQUEST: I need the operational CLI command necessary for such type of device to obtain the information referred in TARGET
    The response will be only a JSON object with the following schema:
    {
        "cli_command":<identified cli command>
    }
    

"""

evaluation_request_prompt_template = """
    A request for information from a certain network device is presented in the field TARGET
    The network device operating system is provided in the field VENDOR_OS
    The actual command required to obtain such information is contained in the field ACTUAL_COMMAND
    The response from a student is contained in STUDENT_RESPONSE
    REQUEST: I need an evaluation on whether the STUDENT_RESPONSE is correct or not, given the ACTUAL_COMMAND provided
    The response for this request will be only a JSON object with the following schema:
    {
        "response_accuracy":<"OK"/"NOK">,
        "rationale":<provida a rationale about why it is OK or what it is NOT OK>
    }
    

"""


#added inline explanation 

def evaluate_vendor_model(benchmark_file,vendor_os,model):
    # Load the benchmark data from the Excel file
    bench_df = load_excel_to_dataframe(benchmark_file)
    # Select only the "target" and vendor_os columns from the DataFrame
    bench_df = bench_df[["target",vendor_os]]
    # Initialize an empty list to store the evaluation results
    evaluation = []

    # Iterate over each row in the DataFrame
    for i,item in bench_df.iterrows():
        # Construct the command request prompt using the target and vendor_os
        command_request_prompt = 'TARGET: '+str(item["target"])+'\nVENDOR_OS: '+vendor_os+'\n'+command_request_prompt_template
        # Get the command response from the model
        command_received = get_completion(command_request_prompt,model=model,DEBUG_MODE=DEBUG_MODE,use_openai=False)

        try:
            # Try to parse the command response as JSON
            command_received = json.loads(command_received)
            # If there is an error in the command response
            if 'error' in command_received:
                # Create an evaluation response with "NOK" accuracy and "LLM error" rationale
                evaluation_response = {
                    "response_accuracy":"NOK",
                    "rationale":"LLM error"
                }
            else:
                # If there is no error, construct the evaluation request prompt using the target, vendor_os, actual command, and student response
                evaluation_request_prompt = 'TARGET: '+str(item["target"])+'\nVENDOR_OS: '+vendor_os+'\nACTUAL_COMMAND: '+item[vendor_os]+'\nSTUDENT_RESPONSE: '+command_received["cli_command"]+'\n'+evaluation_request_prompt_template
                # Get the evaluation response from the baseline model
                evaluation_response = get_completion(evaluation_request_prompt,model=BASELINE_MODEL,DEBUG_MODE=DEBUG_MODE,use_openai=True)
                # Parse the evaluation response as JSON
                evaluation_response = json.loads(evaluation_response)
                # Print a dot to indicate progress
                print('.',end='',flush=True)
                # Add the command response to the evaluation response
                evaluation_response['LLM response']=command_received['cli_command']

        except:
            # If an exception occurs, create an evaluation response with "NOK" accuracy and "Bad LLM response" rationale
            evaluation_response = {
                    "response_accuracy":"NOK",
                    "rationale":"Bad LLM response"
                }
        # Add the question and expected answer to the evaluation response
        evaluation_response['Question']=item['target']
        evaluation_response['Expected answer']=item[vendor_os]
        # Add the evaluation response to the evaluation list
        evaluation.append(evaluation_response)
        
    # Return the evaluation list
    return evaluation



def extract_mark(evaluation):
    total_questions = len(evaluation)
    total_mark = 0
    for answer in evaluation:
        if 'response_accuracy' in answer:
            if answer['response_accuracy']=='OK':
                total_mark += 1
    return total_mark / total_questions * 100 


def run_benchmark(model_list,vendor_list):
    columns = ['vendor_os']+model_list
    benchmark_result = pd.DataFrame(columns=columns)
    benchmark_data = []
    complete_list = []
    for vendor_os in vendor_list:
        for evaluated_model in model_list:
            evaluation_step = {
                'model':evaluated_model,
                'vendor_os':vendor_os
            }
            complete_list.append(evaluation_step)

    evaluation_instance_list = Parallel(n_jobs=-1)(delayed(evaluate_vendor_model)(args.benchmark,step['vendor_os'],step['model']) for step in complete_list)

    for vendor_os in vendor_list:
        print('\n==========================================')
        print('\nEvaluating vendor OS: ',vendor_os)
        vendor_response = [vendor_os]
        for evaluated_model in model_list:
            print('----------------------')
            print('Evaluating LLM model: ',evaluated_model)
            for i in range(len(complete_list)):
                if (complete_list[i]['vendor_os']==vendor_os) and (complete_list[i]['model']==evaluated_model):
                    evaluation_instance = evaluation_instance_list[i] 
            evaluation_mark = extract_mark(evaluation_instance)
            print('     Benchmark result:',evaluation_mark)
            print('----------------------')
            vendor_response.append(evaluation_mark)
            benchmark_entry = {
                "vendor_os":vendor_os,
                "llm":evaluated_model,
                "evaluation":evaluation_instance,
                "mark":evaluation_mark
            }
            benchmark_data.append(benchmark_entry)
        
        benchmark_result.loc[len(benchmark_result)]=vendor_response
    benchmark_result.set_index('vendor_os',inplace=True)
    return benchmark_result,benchmark_data

def dataframe_to_ascii_table(df):
    # Convert DataFrame to ASCII table string
    ascii_table = tabulate(df, headers='keys', tablefmt='grid', showindex="always")
    return ascii_table

def pretty_print_dict(dictionary):
    # Convert dictionary to a formatted string
    formatted_string = json.dumps(dictionary, indent=4)
    return formatted_string


if __name__ == "__main__":
    #run like this python nok_benchmark.py --benchmark path_to_benchmark_file --llms path_to_llms_file --vendors path_to_vendors_file --output path_to_output_file
    
    '''
    
    Replace path_to_benchmark_file, path_to_llms_file, path_to_vendors_file, and path_to_output_file with the paths to your actual files.

    If you don't provide any arguments, the script will use the default values specified in the argparse.ArgumentParser() call:

    Benchmark file: ./benchmarks/benchmark.xlsx
    LLMs file: ./llm_under_test.csv
    Vendors file: ./vendor_os_list.csv
    Output file: ./benchmark_results.txt
    After the script finishes running, it will save the benchmark results to the specified output file and generate a bar chart saved as llm_nok_benchmark_results.jpg in the current directory. The chart will also be displayed in the terminal if your environment supports it.
    
    '''
    
    #or with default values you can run python nok_benchmark.py
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="./benchmarks/benchmark.xlsx")
    parser.add_argument("--llms", default="./llm_under_test.csv")
    parser.add_argument("--vendors", default="./vendor_os_list.csv")
    parser.add_argument('--output', default="./benchmark_results.txt")

    args, _ = parser.parse_known_args()

    model_list = read_single_column_csv(args.llms)
    vendor_list = read_single_column_csv(args.vendors)

    results, data = run_benchmark(model_list,vendor_list)

    final_output = "Benchmark results for: "
    final_output +="\nLLM Model list: "+str(model_list)
    final_output +="\nVendor OS list: "+str(vendor_list)
    final_output +="\n========================================================="
    final_output +="\nSummary results:\n"
    final_output +=dataframe_to_ascii_table(results)
    final_output +="\n========================================================="
    final_output +="\nDetails: \n"
    final_output +=pretty_print_dict(data)

    with open(args.output, 'w') as file:
        file.write(final_output)

    df = results.T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.20  # Width of the bars

    # Positions of the bars on the x-axis
    ind = np.arange(len(df))

    # Creating bars for each vendor
    num_vendors = len(vendor_list)
    for i in range(num_vendors):
        position = width*(i - (num_vendors-1)/2) 
        ax.bar(ind + position, df[vendor_list[i]], width, label=vendor_list[i])

    # Adding labels and title
    ax.set_ylabel('Scores')
    ax.set_xlabel('LLM Models')
    ax.set_title('LLM NOK Benchmark - Network Operational Knowledge. by Device Vendor OS and LLM Model')
    ax.set_xticks(ind)
    ax.set_xticklabels(df['Model'], rotation=30, ha='right')
    ax.legend()

    # Setting the Y-axis range from 0 to 100
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)

    # Save the plot as a JPG file
    plt.savefig('llm_nok_benchmark_results.jpg', format='jpg', dpi=300)  

    # Show the plot
    plt.show()
    





