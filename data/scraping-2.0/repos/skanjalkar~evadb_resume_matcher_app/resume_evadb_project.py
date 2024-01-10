import os
import time
from collections import defaultdict
import sys

import evadb
import openai
from evadb.configuration.constants import EvaDB_INSTALLATION_DIR

from utils import read_text_line, write_dict_to_files

from gptcache import cache



PATH_TO_JOB = "JobDescription/*.pdf"
OUTPUT_DIRECTORY = "./output_directory"
PATH_TO_RESUME = "./JobDescription/job_desc_front_end_engineer.pdf"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
cache.init() # keep a global cache rather than function level cache
cache.set_openai_key() # don't have to but following documentation

def connect_to_database():
    """Connect to the EvaDB database and return a cursor."""
    return evadb.connect().cursor()

def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

# use gpt cache to store the resume summary
# write a funnction for it
def resume_summary_cache(list_data):
    """
    """

    print("Cache loading.....")
    start_time = time.time()
    print("------------------------------------------------------------------------")
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system", "content": "You will be provided with a block of text about someone's resume, and your task is to find out what the person is good at, understand the difficulty of each work they have done so far.",
                'role': 'user', 'content': f'{list_data}'
            }
        ],
    )

def twoprints():
    """ Void function which prints two empty lines
    """
    print()
    print()

def text_summarizer(cursor):
    """Summary of the function: This function will take the job description and summarize it using the hugging face model in built in evadb
    Args: None
    Returns: final_data: list of summarized job description
    """
    cursor.query("""CREATE FUNCTION IF NOT EXISTS TextSummarizer
        TYPE HuggingFace
        TASK 'summarization'
        MODEL 'facebook/bart-large-cnn';""").df()

    cursor.query("""DROP TABLE IF EXISTS JobDescription""").df()
    cursor.query(f"LOAD PDF '{PATH_TO_JOB}' INTO JobDescription").df()

    res = cursor.query("SELECT * FROM JobDescription").df()
    # res = res.tolist()
    # print(res)
    # empty list to store the data
    info = defaultdict(str)
    for _row_id, _data in res.iterrows():
        # get all data for same row id given by _data[_row_id]
        info[_data['_row_id']] += _data['data']
    directory_path = OUTPUT_DIRECTORY
    write_dict_to_files(info, directory_path)


    # return info
    # print(info)
    # drop table if exists
    cursor.query("""DROP TABLE IF EXISTS JobDescriptionSummary""").df()
    cursor.query(f"""LOAD DOCUMENT '{directory_path}/*.txt' INTO JobDescriptionSummary""").execute()
    # print(cursor.query("""SELECT * FROM JobDescriptionSummary""").df())
    # print(info[1])
    # cursor.query(f"""INSERT INTO JobDescriptionSummary (summary) VALUES ('{info[1]}')""").df()
    # for key, val in info.items():
    #     print(cursor.query(f"INSERT INTO JobDescriptionSummary (id, summary) VALUES ({key}, '{val}')").df())

    # print the info from JobDescriptionSummary
    # res = cursor.query("SELECT * FROM JobDescriptionSummary").df()
    # print(res)


    max_id = cursor.query("""SELECT MAX(_row_id) FROM JobDescription""").df()
    # #print max_id
    max_id = max_id['_row_id'].tolist()
    max_id = int(max_id[-1])
    # print(max_id)
    final_data = []
    for i in range(max_id):
        temp_data = cursor.query(f"SELECT TextSummarizer(data) FROM JobDescriptionSummary WHERE _row_id = {i+1}").df()
        final_data.append(temp_data['summary_text'].tolist())

    # for data in final_data:
    #     print("This is the final data", data)
    Text_feat_function_query = f"""CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
            IMPL  '{EvaDB_INSTALLATION_DIR}/functions/sentence_feature_extractor.py';
            """
    cursor.query(Text_feat_function_query).execute()
    cursor.query("""DROP TABLE IF EXISTS FeatureTable""").df()
    cursor.query(
        f"""CREATE TABLE FeatureTable AS
        SELECT SentenceFeatureExtractor(data), data FROM JobDescriptionSummary WHERE _row_id = 1;"""
    ).execute()

    res = cursor.query("""SELECT * FROM FeatureTable""").df()
    print("Final output", res)
    return final_data

def find_match(cursor):
    """Find match: This function will take the resume and job description that is given by hugging face and match them using the gpt 3.5 turbo model from openai
    Args: None
    Returns: None
    """

    cursor.query("DROP TABLE IF EXISTS MyPDFs").df()
    cursor.query(f"LOAD PDF '{PATH_TO_RESUME}' INTO MyPDFs").df()


    data = cursor.query("""
        SELECT data
        FROM MyPDFs
    """).df()
    print(data)
    list_data = data['data'].tolist()
    list_string = ''.join(list_data)


    # completion = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-1106",
    #     messages=[
    #         {"role": "system", "content": "You will be provided with a block of text about someone's resume, and your task is to find out what the person is good at, understand the difficulty of each work they have done so far."},
    #         {"role": "user", "content": f"{list_string}"}
    #     ]
    # )
    # resume_summary = completion.choices[0].message
    # store resume summary so gptcache can use it

    twoprints()

    print("------------------------------------------------------------------------")
    print("Using GPTcache to check if resume summary is cached, if not then cache it")
    resume_summary = resume_summary_cache(list_string)
    print("------------------------------------------------------------------------")

    twoprints()

    replies = {}
    final_data_jobs = text_summarizer(cursor=cursor)
    for i, val in enumerate(final_data_jobs):
        start = time.time()
        print("------------------------------------------------------------------------")
        print("Starting the process of matching the resume with the job description")
        print("------------------------------------------------------------------------")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are now acting as a professional recruiter. You will be given job description and you have to summarize it so that it can be used to match with the resumes."},
                {"role": "user", "content": f"Here is some context about the resume: {resume_summary}"},
                {"role": "user", "content": f"Match the resume with the job description: {val} and only give me a score based on % from 0 to 100 based on how well the resume keywords matches the job description. Don't give me any extra information, only the percentage score."}
            ]
        )
        twoprints()
        end = time.time()
        print("------------------------------------------------------------------------")
        print("Time taken to match each job with resume is: ", end - start)
        print("------------------------------------------------------------------------")
        twoprints()
        replies[i] = completion.choices[0].message
    # print(completion.choices[0].message)
    print(replies)

    # Write benchmark for how much open ai is costing per query
    bill = openai.Billing.retrieve()

    print("------------------------------------------------------------------------")

    print(bill)
    print(bill['data'][0]['cost'])

    print("------------------------------------------------------------------------")



def main():
    start = time.time()
    print("------------------------------------------------------------------------")
    print("Welcome to EvaDB, the database that can understand your data")
    print("------------------------------------------------------------------------")
    print("Starting the process of summarizing the job description")
    cursor = connect_to_database()
    find_match(cursor=cursor)
    end = time.time()
    print("------------------------------------------------------------------------")
    print("Time taken to run the program is:", end - start)
    print("------------------------------------------------------------------------")


if __name__ == '__main__':
    main()