# Imports
import concurrent.futures
from dotenv import load_dotenv
import json
import openai
import os
import pandas as pd
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential
) 
import time

# Setup
retrievalData = './ScrapingResults/Prompts/0-9699/retrievals0-9699.csv' # set accordingly to relevant retrievals csv

promptErrorData = './promptErrors.csv'

load_dotenv()
openai.api_key = os.environ.get('openai.api_key')

# Data Extraction
def extract():
    '''
    Description
        - Wrapper function used to run the data extraction phase.
    Parameters
        - No input parameters.
    Return
        - A dataframe containing each candidate’s name, state, min year, candid, 
        college major, undergraduate institution, highest degree and institution, 
        work history, sources, and ChatGPT confidence. This dataframe is also 
        output to extractions.csv.
    '''
        
    startExtract = time.perf_counter()
    
    # processes retrieval data
    try:
        df = pd.read_csv(retrievalData, index_col = None, encoding = 'latin-1')
        prompts = [
            [prompt, sources, full_name, min_year, state, candid]
            for prompt, sources, full_name, min_year, state, candid in zip(df['ChatGPT Prompt'], df['Sources'], df['Full Name'], df['Min Year'], df['State'], df['Candid'])
        ]
    except:
        print('extract - retrievalData processing error')
    
    # summarizes prompts using ChatGPT API and multithreading
    outputs = []
    promptErrors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chatFeed, prompt): prompt for prompt in prompts}
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                print(f'{output} extract - chatFeed generated an exception: {exc}')
                promptErrors += [output]
                with open('errors.txt', 'a') as f:
                    f.write(f'\n\n{output} extract - chatFeed generated an exception: {exc}')
    doneFeed = time.perf_counter()
    print(f'chatFeed: {doneFeed - startExtract} seconds')
    
    # creates CSVs containing the final results and errors
    extractions = extractCSV(outputs, promptErrors, variant = 'normal')
    doneExtractCSV = time.perf_counter()
    print(f'extractCSV: {doneExtractCSV - doneFeed} seconds')
    
    doneExtract = time.perf_counter()
    print(f'data extraction: {doneExtract - startExtract} seconds')
    return extractions

def extractAgain(attempt = 'first'):
    '''
    Description
        - Wrapper function used to rerun the data extraction phase for candidates 
        who encountered prompt errors.
    Parameters
        - attempt: a string that indicates which type of rerun is being processed. 
        If attempt is set to ‘first’, a new CSV called reruns.csv is created 
        from scratch. If attempt is set to ‘later’, the new results are appended 
        to an already existing reruns.csv. This allows the function to be called 
        multiple times without erasing the progress from previous reruns. attempt 
        is set to ‘first’ by default.
    Return
        - A dataframe containing each rerun candidate’s name, state, min year, 
        candid, college major, undergraduate institution, highest degree and 
        institution, work history, sources, and ChatGPT confidence. This dataframe 
        is also output to reruns.csv.
    '''
        
    # verifies parameters
    assert attempt in ['first', 'later']
        
    startRerun = time.perf_counter()
    
    # processes prompt error data
    df = pd.read_csv(promptErrorData, index_col = None, encoding = 'latin-1')
    prompts = [
        [prompt, sources, full_name, min_year, state, candid]
        for prompt, sources, full_name, min_year, state, candid in zip(df['ChatGPT Prompt'], df['Sources'], df['Full Name'], df['Min Year'], df['State'], df['Candid'])
    ]
    
    # summarizes prompts using ChatGPT API and multithreading
    outputs = []
    promptErrors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(chatFeed, prompt): prompt for prompt in prompts}
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                print(f'{output} rerun - chatFeed generated an exception: {exc}')
                promptErrors += [output]
                with open('errors.txt', 'a') as f:
                    f.write(f'\n\n{output} rerun - chatFeed generated an exception: {exc}')
    doneFeed = time.perf_counter()
    print(f'chatFeed: {doneFeed - startRerun} seconds')
    
    # creates CSVs containing the final results and errors
    reruns = extractCSV(outputs, promptErrors, variant = 'rerun', attempt = attempt)
    doneRerunCSV = time.perf_counter()
    print(f'rerunCSV: {doneRerunCSV - doneFeed} seconds')
    
    doneRerun = time.perf_counter()
    print(f'prompt error rerun: {doneRerun - startRerun} seconds')
    return reruns

@retry(wait = wait_random_exponential(min=45, max=75), stop = stop_after_attempt(10), 
       retry = retry_if_not_exception_type(openai.error.InvalidRequestError),
       before_sleep = lambda _: print("retrying chatFeed"))
def chatFeed(p):
    '''
    Description
        - Uses the ChatGPT API to summarize the biodata from the scraped text 
        and provide a JSON response.
    Parameters
        - p: An array containing the ChatGPT prompt, source URLs, full name, 
        min year, state, and candid of a candidate. The element containing the 
        source URLs is a string array.
    Return
        - An array containing the ChatGPT response, source URLs, full name, min 
        year, state, and candid of a candidate. The element containing the source 
        URLs is a string array.
    '''
    
    # gets ChatGPT response
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        temperature = 0,
        max_tokens = 200,
        messages = [{'role': 'system', 'content': 'Act as a summarizer'},
                    {'role': 'system', 'content': p[0]}]
    )
    output = [response.choices[0].message.content, p[1], p[2], p[3], p[4], p[5]]
    return output

def extractCSV(outputs, promptErrors, variant = 'normal', attempt = 'first'):
    '''
    Description
        - Processes the data gathered in the data extraction stage and converts 
        it into the corresponding pandas dataframes and CSVs. Handles normal 
        responses, prompt errors, and parse errors, which are stored in 
        extractions.csv, promptErrors.csv, and parseErrors.csv, respectively.
    Parameters
        - outputs: a 2D array containing the relevant candidate information for 
        each candidate as the elements in the array. Each element is itself an 
        array containing the ChatGPT response, source URLs, full name, min year, 
        state, and candid of a candidate. The element containing the source URLs 
        is a string array.
        - promptErrors: a 2D array containing all candidates that encountered 
        prompt errors during the chatFeed function. Each element is itself an 
        array containing the ChatGPT prompt, source URLs, full name, min year, 
        state, and candid of a candidate. The element containing the source URLs 
        is a string array.
        - variant: a string that specifies if the outputs are being processed 
        normally or as part of a rerun. If variant is set to ‘normal’, the 
        dataframe containing the final results will be output to extractions.csv. 
        If variant is set to ‘rerun’, the dataframe containing the final results 
        will be output to reruns.csv. variant is set to ‘normal’ by default.
        - attempt: a string that indicates which type of rerun is being processed. 
        If attempt is set to ‘first’, a new CSV called reruns.csv is created from 
        scratch. If attempt is set to ‘later’, the new results are appended to an 
        already existing reruns.csv. This allows the function to be called multiple 
        times without erasing the progress from previous reruns. attempt is set 
        to ‘first’ by default.
    Return
        - A dataframe containing each candidate’s name, state, min year, candid, 
        college major, undergraduate institution, highest degree and institution, 
        work history, sources, and ChatGPT confidence. If variant is set to 
        ‘normal’, this dataframe is also output to extractions.csv. If variant 
        is set to ‘rerun’, this dataframe is instead output to reruns.csv.
    '''
    
    # verifies parameters
    assert variant in ['normal', 'rerun']
    assert attempt in ['first', 'later']
     
    # creates CSV containing prompt errors
    rawPromptErrors = {
        'ChatGPT Prompt': [],
        'Sources': [],
        'Full Name': [],
        'Min Year': [],
        'State': [],
        'Candid': []
    }  
    for error in promptErrors:
        try:
            rawPromptErrors['ChatGPT Prompt'].append(error[0])
            rawPromptErrors['Sources'].append(error[1])
            rawPromptErrors['Full Name'].append(error[2])
            rawPromptErrors['Min Year'].append(error[3])
            rawPromptErrors['State'].append(error[4])
            rawPromptErrors['Candid'].append(error[5])
        except:
            continue
    try:
        promptErrorFrame = pd.DataFrame(rawPromptErrors, columns = ['ChatGPT Prompt', 'Sources',
                                                                    'Full Name', 'Min Year',
                                                                    'State', 'Candid'])
        promptErrorFrame.to_csv('promptErrors.csv')
        print(f'\n{promptErrorFrame.head()}\n{len(promptErrorFrame)} rows\n')
    except:
        print('promptErrorFrame not constructed')
    
    # creates or appends to CSV containing final results
    parseErrors = []
    rawResults = {
        'Name': [],
        'State': [],
        'Min Year': [],
        'Candid': [],
        'College Major': [],
        'Undergraduate Institution': [],
        'Highest Degree and Institution': [],
        'Work History': [],
        'Sources': [],
        'ChatGPT Confidence': []
    }
    
    # parses ChatGPT responses using multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers = None) as executor:
        futures = {executor.submit(parse, output): output for output in outputs}
        for future in concurrent.futures.as_completed(futures):
            output = futures[future]
            try:
                data = future.result()
                if data != None and len(data) == 10: # verifies data exists
                    rawResults['Name'].append(data[0])
                    rawResults['College Major'].append(data[1]) 
                    rawResults['Undergraduate Institution'].append(data[2]) 
                    rawResults['Highest Degree and Institution'].append(data[3])
                    rawResults['Work History'].append(data[4])
                    rawResults['ChatGPT Confidence'].append(data[5])
                    rawResults['Sources'].append(data[6]) 
                    rawResults['Min Year'].append(data[7])
                    rawResults['State'].append(data[8]) 
                    rawResults['Candid'].append(data[9])
                elif data[0] == -1:
                    parseErrors.append(data[1])
            except Exception as exc:
                print(f'{output} extract - parse generated an exception: {exc}')
                with open('errors.txt', 'a') as f:
                    f.write(f'\n\n{output} extract - parse generated an exception: {exc}')
    try:
        df = pd.DataFrame(rawResults, columns = ['Name', 'State', 'Min Year', 'Candid',
                                                 'College Major', 'Undergraduate Institution', 
                                                 'Highest Degree and Institution', 
                                                 'Work History', 'Sources', 'ChatGPT Confidence'])
        if variant == 'normal': 
            df.to_csv('extractions.csv') # stores results to extractions.csv
        elif variant == 'rerun':
            if attempt == 'first':
                df.to_csv('reruns.csv') # stores new results in reruns.csv
            else:
                df.to_csv('reruns.csv', mode = 'a') # appends new results to reruns.csv
        else:
            print('invalid extractCSV variant')
    except:
        df = -1
    
    # creates or appends to CSV containing parse errors
    rawParseErrors = {
        'Parse Error': parseErrors
    }
    try:
        parseErrorFrame = pd.DataFrame(rawParseErrors, columns = ['Parse Error'])
        if variant == 'normal':
            parseErrorFrame.to_csv('parseErrors.csv')
        elif variant == 'rerun': 
            parseErrorFrame.to_csv('parseErrors.csv', mode = 'a')
        else:
            print('invalid extractCSV variant')
        print(f'{parseErrorFrame.head()}\n{len(parseErrorFrame)} rows\n')
    except:
        print('parseErrorFrame not constructed')
        
    if df.empty:
        return rawResults
    else:
        return df

def parse(output):
    '''
    Description
        - Reads the JSON formatted ChatGPT response of a candidate and extracts 
        the full name, college major, undergraduate institution, highest degree 
        and institution, and work history. Candidates whose responses get parsed 
        incorrectly are appended to parseErrors.
    Parameters
        - output: an array containing the ChatGPT response, source URLs, full 
        name, min year, state, and candid of a candidate. The element containing 
        the source URLs is a string array.
    Return
        - If successful, an array containing the full name, college major, 
        undergraduate institution, highest degree and institution, work history, 
        ChatGPT confidence, sources, min year, state, and candid of a candidate. 
        The element containing the source URLs is a string array. If unsuccessful, 
        the return value is an array whose first element is -1.
    '''
        
    data = []
    try:
        d = json.loads(output[0].replace('\n', '')) # splits JSON data
        data = [output[2]]
        data += d.values() # appends ChatGPT response data
        data += [output[1], output[3], output[4], output[5]]
    except:
        return [-1, output]
    return data