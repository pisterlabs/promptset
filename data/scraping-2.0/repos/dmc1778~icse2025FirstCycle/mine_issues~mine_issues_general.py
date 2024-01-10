import json
import os
import re
import requests
import random
import datetime
import time
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from csv import writer
from mine_comments import parse_comment
import csv
import pandas as pd
import replicate
import subprocess
import tiktoken
'''
You need to put four github access token in the following dictionaries
'''
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
    api_key=os.environ.get(".env")
)
url = "https://www.llama2.ai/api"
headers = {
    'Content-Type': 'text/plain'
}
replicate = replicate.Client(api_token='r8_8taKlPQzi3Liw2179bZKZZE7pbRlfS50dSisN')

def get_token_count(string):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    num_tokens = len(encoding.encode(string))

    return num_tokens

def wrap_request_and_send(prompt):
    response = replicate.run(
    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={"prompt": prompt, "max_length":20, "temperature": 0.5 , "top_p": 0.9, "repetition_penalty": 1},)
    return response

def create_prompt(post):
    prompt_ = f"""
    You are an advanced github issue bot analyzer. 
    Your duty is extract at most three meaningful keywords related to GPU/device errors/bugs in the given Github issue wrapped by ####.

    Github issue: ####{post}####

    REMEMBER: Do not explain the bug, just extract three keywords. 
    REMEMBER: Do not start your response with ### Sure, I can help you with that ###.
    REMEMBER: If you can not extract any related keywords, just skip generate any response.
    REMEMBER: Be creative, e.g., you can mix keywords to make them more informative. For example,
    if you extract GPU and error as separate keywords, you can generate a final response as GPU error.

    Please generate the keywords as the following format:

    Keywords: keyword1, keyword2, keyword3.
    """
    return prompt_

def write_csv(data, target, stage=3):
    if not os.path.exists(f'output/keywords/{target}'):
        os.makedirs(f'output/keywords/{target}')

    file_path = f"output/keywords/keywords_{target}.csv"

    with open(file_path, 'a', newline='\n', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data)

TOKEN1 = os.getenv("GIT_TOKEN1")
TOKEN2 = os.getenv("GIT_TOKEN2")
TOKEN3 = os.getenv("GIT_TOKEN3")
TOKEN4 = os.getenv("GIT_TOKEN4")
TARGET = os.getenv('TARGET')
PATTERN_LIST = os.getenv('PATTERN_LIST')

tokens = {
    0: TOKEN1,
    1: TOKEN2,
    2: TOKEN3,
    3: TOKEN4,
}

tokens_status = {
    TOKEN1: True,
    TOKEN2: True,
    TOKEN3: True,
    TOKEN4: True,
}


def match_label(labels):
    label_flag = False
    for l in labels:
        if "bug" in l["name"]:
            label_flag = True
    return label_flag


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session 


retries = 10
now = datetime.datetime.now()

def collect_labels(object):
    labels = []
    for l in object:
        labels.append(l['name'])
    return labels

def completions_with_backoff(prompt, model='gpt-3.5-turbo-1106'):
    response = client.chat.completions.create(
        model=model,
        max_tokens=100,
        messages=[
            {"role": "system", "content": prompt}
        ]
    )
    return response


def get_commits(
    # keyword,
    lib,
    total_issue_counter,
    last_com,
    page_number,
    potential_commits,
    current_token,
):
    page_number += 1
    total_issue_counter += 1

    print("Current page number is: {}".format(page_number))

    headers = {
            "Authorization": f"Bearer {current_token}"
        }

    params = {
            # "q": f"{keyword} in:issue",
            "q": "in:issue is:closed",
            "per_page": 100, 
        }
    if total_issue_counter == 500:
        total_issue_counter = 0
        return
    if page_number == 1:

        issue_link = f"https://api.github.com/repos/{lib[0]}/{lib[1]}/issues"

        response = requests_retry_session().get(
            issue_link,
            params=params,
            headers=headers,
        )
    else:
        response = requests_retry_session().get(
            last_com,
            params=params,
            headers=headers
        )
        link_ = last_com

    if response.status_code != 200:
        tokens_status[current_token] = False
        current_token = select_access_token(current_token)
        response = requests_retry_session().get(
            link_,
            params=params,
            headers=headers
        )

    if response.status_code != 200:
        tokens_status[current_token] = False
        current_token = select_access_token(current_token)
        response = requests_retry_session().get(
            link_,
            params=params,
            headers=headers
        )

    if response.status_code != 200:
        tokens_status[current_token] = False
        current_token = select_access_token(current_token)
        response = requests_retry_session().get(
            link_,
            params=params,
            headers=headers
        )

    if response.status_code != 200:
        tokens_status[current_token] = False
        current_token = select_access_token(current_token)
        response = requests_retry_session().get(
            link_,
            params=params,
            headers=headers
        )

    first_100_commits = json.loads(response.text)

    if len(first_100_commits) == 1:
        return None
    for i, commit in enumerate(first_100_commits):

        if TARGET == 'device':
            pattern = r'(\bERROR: OpenGL error\b|\bfail\b|ROCm fails|ROCm runtime|NCCL error\b|CPU error\b|\bmkldnn error\b|\brocm driver error\b|\bGPUassert\b|\bDLA_RUNTIME Error\b|\bCUDA compilation error\b|\bCUDA MMU fault\b|\bGPU temperature\b|\bVulkan error\b|\bvulkan validation error\b|\bOpenGL Error\b|\bVulkan errors\b|\bGPU version mismatch\b|\bGPU hangs\b|\bdriver issue\b|\bGPU driver issue\b|\bGPU memory issue\b|\bTensorRT error\b|\bGPU compatibility\b|\bcuDNN error\b|\bCUDA error\b|\bGPU support\b|\bGPU error\b|\bGPU utilization\b|\bGPU memory\b)'
        else:
            pattern = r"(FutureWarning:|Warning:|warning:)"
        title_match = False
        body_match = False

        if isinstance(commit["body"], str):
            # if re.findall(r'(from sklearn|import tensorflow as tf|import torch|from mxnet|from mxnet.gluon)', commit['body']):
                    comment_flag = parse_comment(
                        commit["comments_url"], current_token)

                    title_match_keyword = []
                    body_match_keyword = []
                    if re.findall(pattern, commit["title"]):
                        title_match_keyword.append(re.findall(pattern, commit["title"]))
                        title_match = True
                    if re.findall(pattern, commit["body"]):
                        body_match_keyword.append(re.findall(pattern, commit["body"]))
                        body_match = True

                    prompt_ = create_prompt(commit["body"])
                    token_count = get_token_count(prompt_)
                    # time.sleep(5)
                    if token_count <= 4097:
                        res = wrap_request_and_send(prompt_)
                        full_res = ""
                        for item in res:
                            full_res += item
                    else:
                        continue
                    #conversations = completions_with_backoff(prompt_)
                    #res = conversations.choices[0].message.content
                    write_csv([full_res, commit['html_url']],'device')

                    _date = commit["created_at"]
                    sdate = _date.split("-")

                    title_match = body_match = comment_flag = True
                    if title_match or body_match or comment_flag:
                        _date = commit["created_at"]
                        sdate = _date.split("-")
                        print(
                            "Title status: {0}, Body status: {1}, Comment status: {2}".format(
                                title_match, body_match, comment_flag
                            )
                        )

                        data =  [commit["html_url"].split('/')[-3], commit["html_url"],commit["created_at"], 'No version']
                        if not os.path.exists(f"./output/issues/{TARGET}/{lib[1]}"):
                            os.makedirs(f"./output/issues/{TARGET}/{lib[1]}")
                        with open(
                                f"./output/issues/{TARGET}/{lib[1]}/all_issues.csv",
                                "a",
                                newline="\n",
                            ) as fd:
                                writer_object = csv.writer(fd)
                                writer_object.writerow(data
                                )

        if i == len(first_100_commits) - 1:
            last_com = response.links["next"]["url"]
            # if 'next' in response.links:
            #     last_com = response.links["next"]["url"]
            # else:
            #     return
            potential_commits = []

            get_commits(
                # keyword,
                lib,
                total_issue_counter,
                last_com,
                page_number,
                potential_commits,
                current_token,
            )


def search_comit_data(c, commit_data):
    t = []

    for item in commit_data:
        temp = item.split("/")
        t.append("/" + temp[3] + "/" + temp[4] + "/")

    r_prime = c.split("/")
    x = "/" + r_prime[3] + "/" + r_prime[4] + "/"
    if any(x in s for s in t):
        return True
    else:
        return False


def select_access_token(current_token):
    x = ""
    if all(value == False for value in tokens_status.values()):
        for k, v in tokens_status.items():
            tokens_status[k] = True

    for k, v in tokens.items():
        if tokens_status[v] != False:
            x = v
            break
    current_token = x
    return current_token


def main():
    total_issue_counter = 0
    current_token = tokens[0]
    libs = [['tensorflow', 'tensorflow']]
    for lib in libs:
        issue_link = f"https://api.github.com/repos/{lib[0]}/{lib[1]}/issues"
    # for keyword in PATTERN_LIST.split(','):
        print(f"I am working on {lib}")
        headers = {
            "Authorization": f"Bearer {current_token}"
        }

        params = {
            #"q": f"{keyword} in:issue is:closed",
            "q": "in:issue is:closed",
            "per_page": 100,
        }

        response = requests.get(issue_link, headers=headers, params=params)

        if response.status_code != 200:
            tokens_status[current_token] = False
            current_token = select_access_token(current_token)
            response = requests_retry_session().get(
                issue_link,
                params=params,
                headers=headers
            )
            
        if response.status_code != 200:
            tokens_status[current_token] = False
            current_token = select_access_token(current_token)
            response = requests_retry_session().get(
                issue_link,
                params=params,
                headers=headers
            )
            
            
        if response.status_code != 200:
            tokens_status[current_token] = False
            current_token = select_access_token(current_token)
            response = requests_retry_session().get(
                issue_link,
                params=params,
                headers=headers
            )
            

        if response.status_code != 200:
            tokens_status[current_token] = False
            current_token = select_access_token(current_token)
            response = requests_retry_session().get(
                issue_link,
                params=params,
                headers=headers
            )
            
            
        response_text = json.loads(response.text)
        page_number = 0
        if len(response_text) >= 100:
            last_com = response.links["last"]["url"]
            get_commits(
                    # keyword,
                    lib,
                    total_issue_counter,
                    last_com,
                    page_number,
                    response_text,
                    current_token,
                )
        else:
            if TARGET == 'device':
                pattern = r'(\bERROR: OpenGL error\b|\bfail\b|ROCm fails|ROCm runtime|NCCL error\b|CPU error\b|\bmkldnn error\b|\brocm driver error\b|\bGPUassert\b|\bDLA_RUNTIME Error\b|\bCUDA compilation error\b|\bCUDA MMU fault\b|\bGPU temperature\b|\bVulkan error\b|\bvulkan validation error\b|\bOpenGL Error\b|\bVulkan errors\b|\bGPU version mismatch\b|\bGPU hangs\b|\bdriver issue\b|\bGPU driver issue\b|\bGPU memory issue\b|\bTensorRT error\b|\bGPU compatibility\b|\bcuDNN error\b|\bCUDA error\b|\bGPU support\b|\bGPU error\b|\bGPU utilization\b|\bGPU memory\b)'
            else:
                pattern = r"(FutureWarning:|Warning:|warning:)"

            title_match = False
            body_match = False

            for issue in response_text:
                #if re.findall(r'(from sklearn|import tensorflow as tf|import torch|from mxnet|from mxnet.gluon)', issue['body']):
                    comment_flag = parse_comment(issue["comments_url"], current_token)
                    
                    if re.findall(pattern, issue["title"]):
                        title_match = True
                    if re.findall(pattern, issue["body"]):
                        body_match = True

                    _date = issue["created_at"]
                    sdate = _date.split("-")
                    print(sdate[0])
                    title_match = body_match = comment_flag = True
                    if title_match or body_match or comment_flag:
                        _date = issue["created_at"]
                        sdate = _date.split("-")
                        if not os.path.exists(f"./output/issues/{TARGET}/{lib[1]}"):
                            os.makedirs(f"./output/issues/{TARGET}/{lib[1]}")
                        with open(f"./output/issues/{TARGET}/{lib[1]}/all_issues.csv","a",newline="\n",) as fd:
                            writer_object = csv.writer(fd)
                            writer_object.writerow([issue["html_url"].split('/')[-3],issue["html_url"],issue["created_at"],])
            potential_commits = []

if __name__ == "__main__":
    main()
