# Author: Scott Spillias
# Email: scott.spillias@csiro.au

## Import Packages
import os
import xlrd
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import re
import numpy as np
import time
import random
import openai
import string
from dotenv import load_dotenv

load_dotenv()

openAI_key = os.environ.get("OPENAI_KEY")
model_to_use = os.environ.get("MODEL_TO_USE")
n_retries = int(os.environ.get("N_RETRIES"))
temperature = int(os.environ.get("TEMPERATURE"))
save_note = os.environ.get("SAVE_NOTE")
rand_seed = os.environ.get("RAND_SEED")
topic = os.environ.get("TOPIC")
Screening_Criteria=os.environ.get("SCREENING_CRITERIA")
skip_criteria = os.environ.get("SKIP_CRITERIA")
screen_name = os.environ.get("SCREEN_NAME")
proj_location = os.environ.get("PROJ_LOCATION")
debug = os.environ.get("DEBUG")

ScreeningCriteria = Screening_Criteria.split(";")

responses = 'Yes or No or Maybe' # included in prompt; what should the decisions be?
choices = responses.split(' or ') # used to ensure consistency in output format.

def generate_text(openAI_key, prompt, n_reviewers, model_to_use):
    openai.api_key = openAI_key
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    failed = True
    attempt = 0
    while failed:
        attempt += 1
        if attempt < n_retries:
            try:
                completions = openai.ChatCompletion.create(
                    model=model_to_use,
                    messages=messages,
                    max_tokens=512,
                # logprobs = logprobs,
                    n=n_reviewers,
                    stop=None,
                    temperature=temperature)
                failed = False
            except:
                print('Connection Error - Retrying')
                time.sleep(1*2^attempt)     
        else:    
            continue
   # message = completions.choices.message['content']
    return completions

def get_data(prompt,content,n_agents,SC_num,info_all,paper_num):
    prompt = prompt + "\n\n" + content + '\n\n'
    #print(prompt)
    ## Call OpenAI
    assessments = []
    initial_decisions = []
    final_decisions = []
    decisions = []
    conflicts = []
    rand_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
    for r in range(0,n_agents):
        bad_answer = True
        while bad_answer: # Force a correctly formatted response.
            # Add in random string for uniqueness
            if rand_seed:
                rand_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
            else:
                rand_string = rand_string
            print(rand_string)
            new_prompt = f"Ignore this string: {rand_string}" + "\n\n" + prompt
            #print(new_prompt)
            # Call OpenAI
            assessment = generate_text(openAI_key, new_prompt, 1, model_to_use).choices[0].message['content']
            print("\n**** Response ****\n")
            print(assessment)
            assessments.append(assessment)
        # Parse output
            try: ## This try block will fail if the format of the response is not consistent, forcing a re-call to OpenAI
                if 'Final Response' in assessment:
                    SC_elements = assessment.split("Final Response")
                    if 'Initial Response' in SC_elements[0]:
                        thoughts = SC_elements[0].strip().split('Initial Response:')[1].replace(": ","")
                    else:
                        thoughts = SC_elements[0].strip().split('SC:')[1].replace(": ","")
                    if ';' in SC_elements[1]:
                        final = SC_elements[1].split(";")
                    else: 
                        final = SC_elements[1].split(",", 1)
                    decision = final[0].strip().replace(": ","")
                    rationale = final[1].strip().replace(": ","")
                else:
                    final = assessment.split(";")
                    decision = final[0].strip().replace("SC: ","")
                    rationale = final[1].strip().replace("SC: ","")
                    thoughts = ""
                if not(any(choice in decision for choice in choices)):
                    continue
                print(ScreeningCriteria[int(SC_num)-1])
                col_decision = "Accept - SC" + f"{SC_num}: " + ScreeningCriteria[int(SC_num)-1]
                col_rationale = "Rationale - SC" + f"{SC_num}: " + ScreeningCriteria[int(SC_num)-1]
                col_thoughts = "Thoughts - SC" + f"{SC_num}: " + ScreeningCriteria[int(SC_num)-1]
                bad_answer = False  
            except:
                print("Bad Parsing...Trying again...")
        # Store outputs
        info_all[r].at[paper_num,col_decision] = decision
        info_all[r].at[paper_num,col_rationale] = rationale
        info_all[r].at[paper_num,col_thoughts] = thoughts
        initial_decision = ["Yes","No"]["No;" in thoughts]
        #print(initial_decision)
        final_decision = ["Yes","No"]["No" in decision]
       # print(final_decision)
       # Show if an AI agent 'changed its mind'
        conflicted = (initial_decision != final_decision)
        conflicts.append(conflicted)
       # print(conflicted)
        initial_decisions.append(final_decision)
        final_decisions.append(initial_decision)
    return assessments, initial_decisions, final_decisions, conflicts

# Create Prompt to LLM
def add_criteria(Criteria):
    base_prompt = "You are a reviewer for a research project and have been asked to assess whether the "\
             "given paper Title and Abstract meets the following Screening Criteria (SC)."\
                 " In assessing, do not re-interpret the SC, simply assess the SC at face value.\n"\
             "We are only interested in papers that strictly meet the SC.\n"\
             "If not enough information is available, be inclusive as we can follow-up at a later stage."
    base_prompt += '\n\n' + f'SC: ' + Criteria 
    base_prompt += '\n\n' + f"Task: Given the following Title and Abstract, respond"\
            f" to the Screening Criteria (SC) with the following elements, "\
            "Initial Response, Reflection on Initial Response, and Final Response."\
            " Here is an example of how your response should look:\n"\
            "Format: \n"\
            "SC -\n"\
            "Initial Response: Only respond with a Yes or No; Short explanation as rationale.\n"\
            "Reflection: Is the Initial Response correct? Be concise.\n"\
            "Final Response: Strictly only respond with a Yes or No; Short explanation based on reflection. "\
            "\nInitial Response and Final Response should consist of "\
            f"only a {responses} "\
            "followed by a semicolon and a single sentence explanation for your reasoning. Like this: "\
            "\nSC: Final Response; One sentence of reasoning."
    return base_prompt

def save_results(screen_name,info_all):
    if debug:
        new_proj_location = proj_location + "/debug"
    else:
        new_proj_location = proj_location + '/Output'
    if not os.path.exists(new_proj_location):
        os.makedirs(new_proj_location)
    file_path = new_proj_location + '/2a_' + screen_name +'_screen-summary'
    try:
        summary_decisions_new.to_csv(file_path + '.csv', encoding='utf-8', index=True)
    except:
        print("Couldn't Save...is file open?")
    for reviewer in range(len(info_all)):
        index = 1
        file_path = new_proj_location + '/2_' + screen_name +'_screened_' + save_note + "-" + str(index)
        while os.path.isfile(file_path + '.csv'):
            file_path = file_path.split("-")[0] + "-" + str(index)
            index += 1
        print("Saving at " + file_path + '.csv')
        try:
            info_all[reviewer].to_csv(file_path + '.csv', encoding='utf-8', index=True)
        except:
            info_all[reviewer].to_csv(file_path + 'e.csv', encoding='utf-8', index=True)


def main():
    print(topic)
    ## Set-up Screening Run
    excel_sheet = '1_' + screen_name + '.xls'

    papers = pd.read_excel(proj_location + '/' +  'Input' + '/' + excel_sheet).replace(np.nan, '') 
    
    if debug: 
        n_studies = int(os.environ.get("DEBUG_N"))
    else:
        n_studies = len(papers)

    decision_numeric = {'Yes': 2, 'No': 0, 'Maybe': 2} # How should each response be 'counted' when assessing inclusion.

     # Begin Screening

    info = papers[['Title','Abstract']]
    info = info[0:n_studies] # For Debugging
    print('\nAssessing ' + str(len(info)) + ' Papers')
    info[f"Accept"] = "NA"    
    n_agents = int(os.environ.get("N_AGENTS"))
    info_all = [info.copy() for _ in range(n_agents)]
    summary_decisions = info


    # Iteratively move through list of Title and Abstracts
    restart_index = int(os.environ.get("RESTART_INDEX"))
    for paper_num in range(restart_index,len(info[f"Title"].values)): 
        if paper_num % 10 == 0: # Save intermediate results in case of disconnection or other failure.
            print('Saving Intermediate Results...')
            summary_decisions_new = pd.concat([summary_decisions,info_all[0].filter(like=f'Deliberation - SC')], axis = 1)
            if debug:
                    new_proj_location = proj_location + "/debug"
            else:
                new_proj_location = proj_location
            file_path = new_proj_location + '/2a_' + screen_name +'_screen-summary'
            try:
                summary_decisions_new.to_csv(file_path + '.csv', encoding='utf-8', index=True)
            except:
                print("Couldn't Save...is file open?")
    # Print and build base prompt
        print('\nPaper Number: ' + str(paper_num))
        title = info[f"Title"].values[paper_num]
        abstract = info[f"Abstract"].values[paper_num]
        if "No Abstract" in abstract:
            print(abstract)
            print("Skipping to next paper...")
            summary_decisions.at[paper_num,'Accept'] = 'Maybe'
            continue
        content = "Title: " + title + "\n\nAbstract: " + abstract
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(content)
        SC_num = 1
    # Iterate over screening criteria 
        for Criteria in ScreeningCriteria:
            prompt = add_criteria(Criteria)
            #print(prompt)
    ######################
    ######################
            assessments,initial_decisions,final_decisions,conflicts = get_data(prompt,content,n_agents,SC_num,info_all,paper_num) #  Call OpenAI here  #
    ######################
    ######################
            print("\nInitial Decisions: ")
            print(initial_decisions)
            print("\nFinal Decisions: ")
            print(final_decisions)
            print("\nConflicts: ")
            print(conflicts)
            converted_decisions = [decision_numeric.get(element, element) for element in (initial_decisions + final_decisions)] 
            converted_decisions = [element for element in converted_decisions if not isinstance(element, str)]
    # Skip subsequent screening criteria in event of a full rejection across all AI agents.
            if sum(converted_decisions) == 0:
                print("Rejected at SC: " + str(SC_num))
                summary_decisions.at[paper_num,'Accept'] = 'No'
                if skip_criteria and not any(conflicts):
                    break

            SC_num += 1
    # If the paper hasn't been rejected by now, accept it.
        if summary_decisions.loc[paper_num,'Accept'] == "NA":
            summary_decisions.at[paper_num,'Accept'] = 'Yes'
    # End Iterating over articles. 

    summary_decisions_new = summary_decisions
    # Save results
    save_results(screen_name,info_all)


if __name__ == "__main__":
    main()






