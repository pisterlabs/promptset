#!/usr/bin/env python3
import openai
import numpy as np
import configparser
import requests
import re
import argparse
import pathlib
import os

config = configparser.ConfigParser()
config.read('.config')


openai.api_key = config.get('open_api_key', 'key')
ss_url = 'http://api.semanticscholar.org/graph/v1/paper/search?'
# QUERY PARAMETERS(https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers)
fields = (
    'paperId', 'title', 'venue','authors', 'abstract', 'year', 'externalIds', 'influentialCitationCount',
    'citationCount', 'isOpenAccess', 'openAccessPdf', 'fieldsOfStudy'
)

query_list = ('spherical+camera+estimation')

# year range will be randomly chosen from here
# range_classic = np.arange(1990, 2025, 10)


def summarize_paper(paper):
    system = """
    論文を以下の制約に従って要約して出力してください。

    [制約]
    タイトルは日本語で書く
    要点は3つにまとめる


    [出力]
    タイトルの日本語訳

    ・要点1
    ・要点2
    ・要点3
    """

    text = f"title: {paper.title}\nbody: {paper.summary}"
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': text}
                ],
                temperature=0.2,
            )

    summary = response['choices'][0]['message']['content']
    date_str = paper.published.strftime("%Y-%m-%d %H:%M:%S")
    message = f"発行日: {date_str}\n{paper.entry_id}\n{paper.title}\n{summary}\n"
    return message

def generate_url(ss_url_, query_list_, fields_, limit_, offset_):
    all_fields = ''
    for field in fields_:
        all_fields += field + ','
    url = ss_url_ + 'query=' + query_list_ + '&fields=' + all_fields[:-1] + '&limit=' + str(limit_) + '&offset=' + str(offset_)
    return url

def split_papers(data_):
    pattern = '(\"paperId[\s\S]*?}]})'
    papers = re.findall(pattern, data_)
    return papers

def get_paper_info(paper_, fields_):
    info = {}
    for field in fields_:
        pattern = '\"' + field + '\":\"([\s\S]*?)\"'
        info[field] = re.findall(pattern, paper_)
    return info
    
#def get_semanticscholar(query: str, paper_all_numb: int = 5, paper_select_numb: int = 3):


def post_obsidian(message:str, file_path:str, file_name:str):
    dout = pathlib.Path(file_path)
    filename = '{}.md'.format(file_name)
    mdfile_path = os.path.join(dout, filename)

    with open(mdfile_path, 'at') as f:
        f.write(message)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='--paper_id is arxiv paper id')

    parser.add_argument("-i", "--paper_id", type=str, default="None", help="arxiv paper id")
    parser.add_argument('--obsidian', default='/mnt/c/Users/takuo/OneDrive/ドキュメント/Obsidian Vault/paperbank/', help='where is your Obsidian root.')
    args = parser.parse_args()

    if args.paper_id != "None":
        print(summarize_paper(paper[0]))
    else:
        
        for i, paper in enumerate(paper_list):
            try:
                print(str(i+1) + '本目の論文')
                post_obsidian(summarize_paper(paper)+'\n'*3, args.obsidian, "test")
                #print(summarize_paper(paper))
            except:
                print('error')
