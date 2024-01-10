import io
import os
import boto3
import json
import csv
import pandas as pd
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
from apify_client import ApifyClient
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

# TODO: Automatically create folder for each new inference run
# TODO: Output the summary into it's own file in the same folder

# Initialize Environment variables
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')
APIFY_TOKEN = os.getenv('APIFY_TOKEN')
s3_client = boto3.client("s3")
print(AWS_S3_BUCKET)

# Initialize LLM
claude_v1_model_id = 'anthropic.claude-instant-v1'
claude_v2_model_id = 'anthropic.claude-v2'
llm = Bedrock(model_id=claude_v1_model_id, model_kwargs={
              'max_tokens_to_sample': 8000})


def split_job(job_title):
    job_name_split = job_title.split(' ')
    job_folder_name = ""
    for index, word in enumerate(job_name_split):
        # don't add _ if it's the last word
        if index == len(job_name_split) - 1:
            job_folder_name += word
        else:
            job_folder_name += word + "_"
    return job_folder_name
# Functionality for apify


def indeed_scrapper(AWS_S3_BUCKET, APIFY_TOKEN, s3_client, job_title="data engineer"):
    client = ApifyClient(APIFY_TOKEN)

    run_input = {
        "position": job_title,
        "country": "US",
        "location": "remote",
        "maxItems": 200,
        "parseCompanyDetails": False,
        "saveOnlyUniqueItems": True,
        "followApplyRedirects": True,
        "maxConcurrency": 5,
    }

    actor_call = client.actor(
        'misceres/indeed-scraper').call(run_input=run_input)

    dataset_items = client.dataset(
        actor_call['defaultDatasetId']).download_items(item_format="csv")

    # Convert the timezone to local
    tz = actor_call['finishedAt'].replace(
        tzinfo=timezone.utc).astimezone(tz=None)
    formatted_date = tz.strftime("%Y-%m-%d_%H-%M-%S")
    # Create filename based on when the scraper finished
    file_name = "dataset_indeed-scraper_"+formatted_date+".csv"
    print(file_name)
    print(dataset_items.decode('utf-8'))

    # Write to the csv
    new_data = dataset_items.decode('utf-8')
    file = open(file_name, "w", newline='')
    file.write(new_data)
    new_df = pd.read_csv(file_name)
    new_df
    # Upload to S3
    job_folder = split_job(job_title)
    s3_filename = f"bronze/raw_indeed_jobs/"+job_folder+"/"+file_name
    s3_client.upload_file(file_name, AWS_S3_BUCKET, s3_filename)





# Master prompt templates
template_master = PromptTemplate.from_template("""
Act as an expert formatter. You format based on the given format. Skip the preamble.
I will provide you with a combined list of job skills and responsibilities text for Data Engineering taken from multiple job postings, output the top 5 skills AND technologies that appear most often across the entire text, do not include skills such as Data Engineering or ETL processes, be specific in the following format: <topskills>1. Skill 2. Skill ...</topskills> <toptech>1. Tech 2. Tech .... </toptech>.
<dataengineeringtext>{text}</dataengineeringtext>
""")

# ### List out the Raw indeed datasets in the bronze bucket, the append them to be downloaded later
def process_indeed_job(AWS_S3_BUCKET, s3_client, job_title):
    job_folder = split_job(job_title)
    response = s3_client.list_objects(
        Bucket=AWS_S3_BUCKET, Prefix=f"bronze/raw_indeed_jobs/{job_folder}/dataset")
    contents = response["Contents"]
    datasets = []
    for item in contents:
        datasets.append(item["Key"])
        print(item["Key"])
    print(datasets)

# Get the dataset CSVs, convert them to dataframe, then store them in local array
# Get the data from bronze bucket and store it in array to be worked on
    dataframes = []
    for data in datasets:
        job_object = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=data)
        status = job_object.get("ResponseMetadata", {}).get("HTTPStatusCode")
        jobs_df = pd.core.frame.DataFrame
        if status == 200:
            print("Successfully got desired object")
            jobs_df = pd.read_csv(job_object.get("Body"))
            dataframes.append(jobs_df)
        else:
            print("Unable to get object, something wrong")


# Filter out NaN values from the datasets then save them
    salary_filtered_frames = []
    for frame in dataframes:
        filtered_salary = frame['salary'].notnull()
        filtered_curr = frame[filtered_salary]
        salary_filtered_frames.append(filtered_curr)

# Merge all of the dataframes, drop the duplicates that appear
# Final frame should represent correct number of uniques
    merged_frame = pd.DataFrame
    for index, frame in enumerate(salary_filtered_frames):
        if index == 0:
            merged_frame = frame
        else:
            new_frame = merged_frame.merge(frame, how="outer")
            merged_frame = new_frame
    merged_frame.reset_index()
    final_frame = merged_frame.drop_duplicates(subset=['id'])
    final_frame.reset_index()
    return final_frame


def create_summaries_from_dataframe(final_frame, llm, template, path):
    file_name = 'claude_v1_df_final_row_'
    for index, row in final_frame.iterrows():
        description = row['description']
        url = row['url']
        file = open(f"{path}{file_name}{index}.txt", "w", encoding="utf-8")
        prompt = template.format(job=description)
        output = llm.predict(prompt)
        file.write(output)
        file.write(f"\n\n{url}")
        file.close()
# Combine all of the documents from a path


def combine_files_string(path, combined_file_name):
    docs = os.listdir(path)
    combined_documents_string = ""
    for doc in docs:
        doc_file = open(path+doc, "r", encoding="utf-8")
        doc_text = doc_file.read()
        combined_documents_string += f"\n\n{doc_text}"
        doc_file.close()
    # Write to new file
    combined_file = open(combined_file_name, "w", encoding="utf-8")
    combined_file.write(combined_documents_string)
    combined_file.close()
    # return combined_documents_string


def create_combined_file_template(llm, combined_file_name, job_title):
    huge_file_read = open(combined_file_name, "r", encoding="utf-8")
    huge_file_text = huge_file_read.read()
    template = PromptTemplate.from_template("""
Act as an expert formatter. You format based on the given format. Skip the preamble.
I will provide you with job skills and tech text for {job_title}, output the top 5 skills AND technologies in the following format: <topskills>1. Skill 2. Skill ...</topskills> <toptech>1. Tech 2. Tech .... </toptech>.
<{job_title} text>{text}</{job_title} text>
""")
    prompt = template.format(text=huge_file_text, job_title=job_title)
    output = llm.predict(prompt)
    print(output)
    huge_file_read.close()



job_title = "data engineer"
job_folder = split_job(job_title)
template = PromptTemplate.from_template("""
Extract specific skills and tech from the following job description: {job}
""")
# Take in a Dataframe, llm, and template and create job summaries
path = 'skills-res-v6-dataengineer-day-5/'
combined_file_name = f"job_summaries_massive_{job_folder}_10_25_23.txt"
indeed_scrapper(AWS_S3_BUCKET, APIFY_TOKEN,
                s3_client, job_title=job_title)
print("Getting files from S3")
final_frame = process_indeed_job(AWS_S3_BUCKET, s3_client, job_title)
print("Creating job summaries")
create_summaries_from_dataframe(final_frame, llm, template, path)
print("Combining files")
combine_files_string(path, combined_file_name)
print("Creating large insights")
create_combined_file_template(llm, combined_file_name, job_title)


###################### Codeium practice examples #################
# read dataset_indeed-scraper_2023-10-23_07-07-15.csv
# Read the dataset
""" df = pd.read_csv("dataset_indeed-scraper_2023-10-23_07-07-15.csv")

# remove NaN values from salary column
df.dropna(subset=['salary'], inplace=True)
len(df)
df

# clean up index of df
df.reset_index(drop=True, inplace=True)
len(df) """
