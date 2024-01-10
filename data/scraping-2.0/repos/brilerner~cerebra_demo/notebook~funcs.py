from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import time
from datetime import datetime
from ast import literal_eval
import openai

ENTRY_PATH = "journal_entries.csv"
DATA_PATH = "data.csv"

# improve try/except for journal

class Cerebra:

    def __init__(self, current_day=None):
        self.DATA_PATH = DATA_PATH
        self.ENTRY_PATH = ENTRY_PATH
        self.current_day = current_day # for testing
        if not self.current_day:
            self.current_day = get_current_date()

        self.agent = self.start_agent()

    def input_prompt(self, prompt):
        try:
            journal_kw = ["journal", "entry"]
            query_kw = ["query", "question"]
            if any(w in prompt for w in journal_kw):
                self.record_journal_entry(prompt, self.current_day)
                response = "Thank you. Your Cerebra Journal has been updated."
            elif any(w in prompt for w in query_kw):
                response = self.run_query(prompt)

            return response
        except:
            # raise
            response = "I'm sorry. I was not able to handle that. Please try again."

            return response

    def run_query(self, query):
        prompt = get_query_prompt(query)
        output = self.agent.run(prompt)
        return output

    def record_journal_entry(self, entry, current_day):

        # save journal entry
        new_journal_entry_df = pd.DataFrame([{"d":current_day, "journal_entry":f'"{entry}"'}])
        update_or_create_df(self.ENTRY_PATH, new_journal_entry_df)

        # set up prompt
        prompt = get_journal_prompt(entry)

        # get response
        output = get_completion(prompt)

        # update data
        current_df = convert_journal_output(output, current_day)
        update_or_create_df(self.DATA_PATH, current_df)

    def start_agent(self):

        agent = create_csv_agent(
            OpenAI(
                model = 'text-davinci-003', # default
                temperature=0
            ), 
            self.DATA_PATH, 
            verbose=True
        )

        return agent


def update_or_create_df(csv_path, new_data):
    try:
        # Try to read the existing CSV
        existing_df = pd.read_csv(csv_path)
        # Concatenate the new DataFrame with the existing one
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    except FileNotFoundError:
        # If the CSV does not exist, we'll just use the new data
        updated_df = new_data
    
    # Save the updated DataFrame back to CSV
    updated_df.to_csv(csv_path, index=False)
    # return updated_df




def get_current_date():

# Get the current date
    current_date = datetime.now()

    # Format the date as '5-Nov'
    formatted_date = current_date.strftime('%-d-%b')

    return formatted_date



def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def convert_journal_output(output, current_date):

    resp = literal_eval(output)
    rows = []
    for label, times in resp.items():
            row = {'d':current_date, 'data_type':'q', 'label': label}
            row.update(times)
            rows.append(row)


    w_resp = {'bpm': {'12am-8am':70, '8am-4pm':100, '4pm-12am':90},
              'calories':{'12am-8am':50, '8am-4pm':200, '4pm-12am':100},
              'sleep':{'12am-8am':4, '8am-4pm':0, '4pm-12am':0} }
    for label, times in w_resp.items():
        row = {'d':current_date, 'data_type':'w', 'label': label}
        row.update(times)
        rows.append(row)

    current_df = pd.DataFrame(rows)
    return current_df


def get_journal_prompt(journal_entry):

    pre_prompt = "Here is the way I would you like you \
    to extract the labels from this journal entry. The labels in the following example \
    are exactly the ones that I would like you to extract. \
    Please note that the actual word that the label contains \
    may not be present in the journal entry, we care about the meaning; \
    i.e. it might be phrase slighly differently but that's ok.  \
    The desired output is a dictionary where each key is a label \
    and the value is a nested dictionary, for which each key is a \
    time of day ('12am-8am', '8am-4pm', '4pm-12am') and the value is \
    either 1 or 0 (binary for whether that label occurred during that time of the day). \
    The time of day keys correlate as follows: '12am-8am' is the time of day between 12 am\
    and 8 am, and so on. The only labels that should be used are 'headache', 'worked_out', and 'shower'.\
    These exact labels should be used in the output!\
    Here is an example format (numbers are made up):\n\
    '{'headache': {'12am-8am':0, '8am-4pm':0, '4pm-12am':1},'worked_out':{'12am-8am':1, '8am-4pm':1, '4pm-12am':1},'shower':{'12am-8am':0, '8am-4pm':0, '4pm-12am':1} }'\n\
    Please give me the ouput.\n\nJournal Entry:\n"

    journal_entry = "I woke up this morning with a pounding headache. I took some medicine and it helped alleviate the pain a bit. After breakfast I went for a 2 mile jog in the park near my house. The fresh air and exercise helped me feel refreshed. When I got home, I took a hot shower to relax my muscles. I spent the afternoon meal prepping and reading out on the patio. For dinner I cooked up some chicken and veggies. I watched a little TV before an early bedtime."

    prompt = pre_prompt + journal_entry + "\n\nOutput:\n"

    return prompt

# query = "How many times did I have a headache?"
def get_query_prompt(query):

    pre_prompt = "Here is a breakdown of the columns present in this data: \n \
    - d: the day the label was extracted \n \
    - label: the label extracted \n \
    - 12am-8am: the first time period of the day \n \
    - 8am-4pm: the second time period of the day \n \
    - 4pm-12am: the third and final time period of the day \n \
    When information about a label is requested in the query, please use all time period columns."


    prompt = pre_prompt + "\n\n" + "Query: " + query

    return prompt