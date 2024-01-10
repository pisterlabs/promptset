import argparse
import json
import os
import time

import requests
import openai
import tqdm
import re

import shortuuid
import logging
import glob

from datetime import datetime

logging.basicConfig(level=logging.INFO)

logFormatter = logging.Formatter("%(asctime)s %(message)s")
logger = logging.getLogger()

def get_json_list_dir(dir_path):
    files = glob.glob(os.path.expanduser(dir_path) + "/*.jsonl")

    jsonl=[]
    for f in files:
        jsonl.extend(get_json_list(f))

    return jsonl

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            review=json.loads(line, strict=False)
            review["file"]=file_path
            json_list.append(review)
        return json_list

def parse_winner(review):
    try:
        core_pattern="[wW]inner:?[\\w ]*([tT]ie|[12])"

        regexs=[]
        regexs.append("^{}.? *$".format(core_pattern))  
        regexs.append("\n{}.? *$".format(core_pattern))  
        regexs.append("^{}.? *\n".format(core_pattern))  
        regexs.append("^{}.".format(core_pattern))  
        regexs.append("{}.? *$".format(core_pattern))  
        regexs.append("\n{}.? *".format(core_pattern))  
        regexs.append("a? ?([tT]ie|[12]).?$")
        regexs.append("^[wW]inner:?[\\w ]*([nN]either|[nN]one)")
        regexs.append("{}".format(core_pattern))  

        for regex in regexs:
            matches = re.search(regex, review)

            if matches is not None: 
                mg = matches.groups()

                if mg[0].isnumeric():
                    return int(mg[0])
                elif mg[0].lower()=="tie" or mg[0].lower()=="neither" or mg[0].lower()=="none":
                    return 0
                else:
                    return -1
    except Exception as e:
        print(e)
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-r", "--review-dir", default="table/reviews_pairwise")
    args = parser.parse_args()

    reviews_json = get_json_list_dir(args.review_dir)

    # find the ones where the parsed winner does not equal the given winner
    diffcnt=0
    for i in range(len(reviews_json)):
        rev=reviews_json[i]
        winner_parsed=str(parse_winner(rev["text"]))
        winner_actual=str(rev["winner"])

        if winner_parsed != winner_actual:
            print("Review {id} in {file}: parsed {parsed} but acutal is {actual}:\n{txt}".format(
                    id=rev["review_id"],
                    parsed=winner_parsed,
                    actual=winner_actual,
                    txt=rev["text"],
                    file=rev["file"]
                ))
            print("------------------------")
            diffcnt+=1

    print(f"found {diffcnt} differences")

    # find the reviews where the text is long which is where parse_winner usually struggles
    diffcnt=0
    for i in range(len(reviews_json)):
        rev=reviews_json[i]
        if len(rev["text"])>20:
            print("Looong Review {id} in {file}: winner {winner}\n{txt}".format(
                    id=rev["review_id"],
                    txt=rev["text"],
                    winner=str(rev["winner"]),
                    file=rev["file"]
                ))
            print("------------------------")
            diffcnt+=1

    print(f"found {diffcnt} long reviews")

    # find the reviews where the winner is NONE
    diffcnt=0
    for i in range(len(reviews_json)):
        rev=reviews_json[i]
        if rev["winner"] is None:
            os.system("open {}".format(rev["file"]))
            print("Looong Review {id} in {file}: winner {winner}\n{txt}".format(
                    id=rev["review_id"],
                    txt=rev["text"],
                    winner=str(rev["winner"]),
                    file=rev["file"]
                ))
            print("------------------------")
            diffcnt+=1

    print(f"found {diffcnt} NONE reviews")

