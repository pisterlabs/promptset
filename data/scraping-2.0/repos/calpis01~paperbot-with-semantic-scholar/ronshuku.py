#!/usr/bin/env python3
import openai
import configparser
import arxiv
import random
import argparse

config = configparser.ConfigParser()
config.read('.config')

openai.api_key = config.get('open_api_key', 'key')


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

def get_arxiv(query: str, paper_all_numb: int = 5, paper_select_numb: int = 3):
    # search arxiv paper
    result = arxiv.Search(
        query=query,
        max_results=paper_all_numb,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    ).results()

    return random.sample(list(result), k=paper_select_numb)


def get_specific_paper_from_arxiv(arxiv_id: str):
    result = arxiv.Search(id_list=[arxiv_id]).results()

    return list(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='--paper_id is arxiv paper id')

    parser.add_argument("-i", "--paper_id", type=str, default="None", help="arxiv paper id")

    args = parser.parse_args()

    if args.paper_id != "None":
        paper = get_specific_paper_from_arxiv(arxiv_id=args.paper_id)
        print(summarize_paper(paper[0]))
    else:
        paper_list = get_arxiv(query='abs:"spherical camera"', paper_all_numb=100, paper_select_numb=3)
        for i, paper in enumerate(paper_list):
            try:
                print(str(i+1) + '本目の論文')
                print(summarize_paper(paper))
            except:
                print('error')
