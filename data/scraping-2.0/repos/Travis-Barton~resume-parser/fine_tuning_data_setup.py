import os
import re
from typing import Union
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from utils import file_reader
import dotenv
from tqdm import tqdm
from transformers import GPT2Tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

dotenv.load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
embeder = OpenAIEmbeddings()

JSON_TEMPLATE = {
                "messages": [{
                                 "role": "system",
                                 "content": "You are a resume parser. Extract the "},  # the type of data to extract
                             {
                                 "role": "user",
                                 "content": ""},  # the resume as a whole
                             {
                                 "role": "assistant",
                                 "content": ""}]}  # the section extracted

def process_profiles(path: str):
    """
    Process profiles in the specified directory and return a list of JSON objects.
    """
    profiles_data = []
    for dir_path in tqdm(os.listdir(path)):
        aim_profile = ''
        resume = ''
        full_dir_path = os.path.join(path, dir_path)
        for file in os.listdir(full_dir_path):
            if file.startswith('~'):
                continue
            file_path = os.path.join(full_dir_path, file)
            mode = 'r' if file.endswith('.txt') else 'rb'
            with open(file_path, mode) as f:
                try:
                    if file.startswith('AIM P'):
                        aim_profile = file_reader(f)
                    else:
                        resume = file_reader(f)
                except Exception as e:
                    print(f'Error reading {file_path}: {e}')
                    continue
        try:
            summary, skills, experience, education, certifications, awards = parse_aim_resumes(aim_profile)
        except Exception as e:
            print(f'Error parsing {dir_path}: {e}')
            continue
        if not resume:
            continue
        try:
            profile_data = {
                "resume": resume,
                "summary": summary,
                "skills": skills,
                "experience": experience,
                "education": education,
                "certifications": certifications,
                "awards": awards,
            }
            profiles_data.append(profile_data)
        except Exception as e:
            print(f'Error embedding {dir_path}: {e}')
            continue

    return profiles_data


# def parse_aim_resumes(resume: str):
#     """
#     AIM resumes have a specific, consistent format.
#     """
#     splits = ['SUMMARY', "SKILLS AND TECHNOLOGIES", "PROFESSIONAL EXPERIENCE", "EDUCATION", "CERTIFICATIONS"]
#     summary = resume.split(splits[0], 1)[1].split(splits[1], 1)[0]
#     skills_and_tech = resume.split(splits[1], 1)[1].split(splits[2], 1)[0]
#     professional_experience = resume.split(splits[2], 1)[1].split(splits[3], 1)[0]
#     education = resume.split(splits[3], 1)[1].split(splits[4], 1)[0]
#     certifications = resume.split(splits[4], 1)[1]
#
#
#     return summary, skills_and_tech, professional_experience, education, certifications
def parse_aim_resumes(resume: str):
    """
    AIM resumes have a specific, consistent format.
    """

    # Regular expressions for each section
    section_patterns = {
        'summary': r'SUMMARY\s*(.*?)(?=SKILLS AND TECHNOLOG(?:Y|IES)|PROFESSIONAL EXPERIENCE|EDUCATION|CERTIFICATIONS|AWARDS|$)',
        'skills_and_tech': r'SKILLS AND TECHNOLOG(?:Y|IES)\s*(.*?)(?=PROFESSIONAL EXPERIENCE|EDUCATION|CERTIFICATIONS|AWARDS|$)',
        'professional_experience': r'PROFESSIONAL EXPERIENCE\s*(.*?)(?=EDUCATION|CERTIFICATIONS|AWARDS|$)',
        'education': r'EDUCATION\s*(.*?)(?=CERTIFICATIONS|AWARDS|$)',
        'certifications': r'CERTIFICATIONS\s*(.*?)(?=AWARDS|$)',
        'awards': r'AWARDS\s*(.*)'
    }

    # Extract sections using regex
    sections = {}
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, resume, re.DOTALL)
        sections[section_name] = match.group(1).strip() if match else None

    return (sections['summary'], sections['skills_and_tech'], sections['professional_experience'],
            sections['education'], sections['certifications'], sections['awards'])

def format_for_finetune(json_path: Union[str, bytes, os.PathLike], save_path: Union[str, bytes, os.PathLike]):
    """
    Format the data for fine-tuning saving a jsonl file.
    """
    if os.path.exists(save_path):
        os.remove(save_path)
    succeeded = 0
    failed = 0
    with open(json_path, 'r') as f:
        profiles_data = json.load(f)

    for idx, profile_data in enumerate(profiles_data):
        resume = profile_data['resume']
        for key, value in profile_data.items():

            json_format = JSON_TEMPLATE.copy()
            if (key == 'resume') or (not value):
                continue
            json_format['messages'][1]['content'] = resume
            json_format['messages'][2]['content'] = value
            json_format['messages'][0]['content'] = f'You are a resume parser. Extract the {key} section'
            # check if total length is less than 4096
            full_str = str(json_format)
            if len(tokenizer.encode(full_str)) > 4096:
                print(f'Length of {idx} is greater than 4096. Length: {len(tokenizer.encode(full_str))}')
                failed += 1
                del json_format
                continue
            succeeded += 1
            with open(save_path, 'a') as f:
                json.dump(json_format, f)
                f.write('\n')
            del json_format
    print(f'Succeeded: {succeeded}, Failed: {failed}')
    print(f'Ratios: {succeeded / (succeeded + failed)}')


def start_finetuning_job(file_path: Union[str, bytes, os.PathLike]):
    """
    Start a fine-tuning job.
    Current model
    {
      "object": "fine_tuning.job",
      "id": "ftjob-jHaLv1MpjJSs1wRPhxdiYXla",
      "model": "gpt-3.5-turbo-0613",
      "created_at": 1693960013,
      "finished_at": 1693963571,
      "fine_tuned_model": "ft:gpt-3.5-turbo-0613:personal::7vbb2i7t",
      "organization_id": "org-ev4E5NkqvqtGmw8yJ80vok7v",
      "result_files": [
        "file-U99WerWwYyvENhHP0ckn7q1M"
      ],
      "status": "succeeded",
      "validation_file": null,
      "training_file": "file-LdgbGbVzGwXs7pIzdOf62NbU",
      "hyperparameters": {
        "n_epochs": 3
      },
      "trained_tokens": 2265357
    }
    """
    filename = os.path.basename(file_path)

    # check if job is running already
    jobs = openai.FineTuningJob.list()
    if jobs['data']:
        print(f'Job {filename} is already running.')
        return

    data_resp = openai.File.create(
        file=open(file_path, "rb"),
        purpose='fine-tune',
        user_provided_filename=filename
    )
    while data_resp['status'] != 'processed':
        data_resp = openai.File.retrieve(id=data_resp['id'])
    response = openai.FineTuningJob.create(training_file=data_resp['id'], model="gpt-3.5-turbo")
    print(response)
    while response['status'] != 'succeeded':
        response = openai.FineTuningJob.retrieve(id=response['id'])
    return response


if __name__ == '__main__':
    import json
    FINETUNE = True
    # profiles_data = process_profiles(path='profiles')
    # with open('profiles_data.json', 'w') as f:
    #     json.dump(profiles_data, f, indent=4)
    # format_for_finetune(json_path='profiles_data.json', save_path='fine_tuning_data.jsonl')
    if FINETUNE:
        start_finetuning_job(file_path='fine_tuning_data.jsonl')
