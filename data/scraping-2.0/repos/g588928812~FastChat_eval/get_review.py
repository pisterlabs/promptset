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

from datetime import datetime

logging.basicConfig(level=logging.INFO)

logFormatter = logging.Formatter("%(asctime)s %(message)s")
logger = logging.getLogger()

logger_logfn = "log_getReview_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fileHandler = logging.FileHandler("{0}/{1}.log".format("./logs/", logger_logfn))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

MAX_API_RETRY = 5
REQ_TIME_GAP = 10

oaikey = None

def get_eval_OAI(reviewer, prompt: str, max_tokens: int):
    logging.basicConfig(level=logging.INFO)

    assert(oaikey is not None)

    for i in range(MAX_API_RETRY):
        try:
            openai.api_key = oaikey

            completion = openai.ChatCompletion.create(
              model=reviewer["params"]["model"],
              temperature=reviewer["metadata"]["temperature"],
              messages=[
                {"role": "user", 
                "content": prompt}
              ],
              max_tokens=max_tokens,
            )

            content = completion.choices[0].message.content

            # logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def get_eval_OOB(reviewer, prompt: str, max_tokens: int):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        # try:
        params=reviewer["metadata"]

        payload = json.dumps([prompt, params])

        server=reviewer["params"]["oobabooga-server"]
        response = requests.post(f"http://{server}/run/textgen", json={
            "data": [
                payload
            ]
        }).json()

        raw_reply = response["data"][0]
        content = raw_reply[len(prompt):]

        # logger.info(content)
        return content
        # except Exception as e:
        #     logger.error(e)
        #     time.sleep(5)
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def parse_score(review, reviewer):
    try:
        regex = reviewer["score-regex"]
        matches = re.search(regex, review)
        mg = matches.groups()

        return [float(mg[0]), float(mg[1])]
    except Exception as e:
        return [-1, -1]

def parse_winner(review, reviewer):
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

def gen_prompt(reviewer, ques, cat, ans1, ans2):
    prompt_template = reviewer["prompt_templates"][cat] if cat in reviewer["prompt_templates"] else reviewer["prompt_templates"]["default"]

    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2
    )

    return prompt

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line, strict=False))
        f.close()
        return json_list

def import_json(file_path):
    file_path = os.path.expanduser(file_path)
    f = open(file_path)
    data = json.load(f)
    f.close()

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question-file", default="table/question.json")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-rf", "--reviewer-file", default="table/reviewer.json")
    parser.add_argument("-r", "--reviewer", default="gpt-3.5-turbo")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument("-k", "--openaikey", type=str)
    parser.add_argument("-dr", "--do-repetitions", type=int, default=1)
    parser.add_argument("-lc", "--limit-category", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    oaikey = args.openaikey
    question_jsons = import_json(args.question_file)
    reviewer_jsons = import_json(args.reviewer_file)
    answer1_jsons = get_json_list(args.answer_file_list[0].replace(":","_"))
    answer2_jsons = get_json_list(args.answer_file_list[1].replace(":","_"))

    reviewers = list(filter(lambda reviewer_jsons: reviewer_jsons['reviewer_id'] == args.reviewer, reviewer_jsons))

    if(len(reviewers)==0):
        logger.error(f"reviewer {args.reviewer} not found in {args.reviewer_file}")
        quit()
    else:
        reviewer=reviewers[0]

    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))

    logger.info("Reviewing {} vs {}, judged by {}".format(args.answer_file_list[0], args.answer_file_list[1],reviewer["reviewer_id"]))


    for rep in range(args.do_repetitions):
        review_jsons = []
    
        logger.info(f"Doing repetition {rep+1} of {args.do_repetitions}")

        output_fn= f"{args.output_review_file}.jsonl" if rep==0 else f"{args.output_review_file}_{str(rep)}.jsonl"
        if os.path.exists(output_fn):
            logger.info(f"Output file {output_fn} already exists, skipping")
            continue

        for i in question_idx_list:
            assert (
                answer1_jsons[i]["question_id"]
                == question_jsons[i]["question_id"]
                == answer2_jsons[i]["question_id"]
            )

            ques = question_jsons[i]["text"]
            cat = question_jsons[i]["category"]
            ans1 = answer1_jsons[i]["text"]
            ans2 = answer2_jsons[i]["text"]
            model1 = answer1_jsons[i]["model_id"]
            model2 = answer2_jsons[i]["model_id"]

            if args.limit_category is not None:
                logger.info(f"Limiting to category '{args.limit_category}'")
                if cat != args.limit_category:
                    continue

            prompt = gen_prompt(reviewer, ques, cat, ans1, ans2)

            if reviewer["type"] == "OpenAI":
                review = get_eval_OAI( reviewer, prompt, args.max_tokens)
            elif reviewer["type"] == "oobabooga-api" :
                review = get_eval_OOB( reviewer, prompt, args.max_tokens)
            else:            
                logger.error("unknown reviewer type " + reviewer["type"])
                quit()

            winner = parse_winner(review, reviewer)

            review_id = shortuuid.uuid()
            review_jsons.append(
                {
                    "review_id": review_id,
                    "question_id": question_jsons[i]["question_id"],
                    "answer1_id": answer1_jsons[i]["answer_id"],
                    "answer2_id": answer2_jsons[i]["answer_id"],
                    "reviewer_id": reviewer["reviewer_id"],
                    "metadata": reviewer["metadata"],
                    "text": review,
                    "scores": parse_score(review, reviewer),
                    "winner": winner 
                }
            )

            logger.info("Review for question {qid}, {m1} vs. {m2}, reviewer {reviewer}. Winner: {winner}".format(
                qid=question_jsons[i]["question_id"],
                reviewer=reviewer["reviewer_id"],
                review="",
                s1=review_jsons[len(review_jsons)-1]["scores"][0],
                s2=review_jsons[len(review_jsons)-1]["scores"][1],
                m1=answer1_jsons[i]["model_id"],
                m2=answer2_jsons[i]["model_id"],
                winner=winner
                ))


            # print("#### PROMPT " + prompt )


            # To avoid the rate limit set by OpenAI
            # logger.info(f"Waiting for {REQ_TIME_GAP} seconds before sending the next request.")
            time.sleep(1)

        logger.info(f"writing output of {model1} vs {model2} to file {output_fn}")
        with open(f"{output_fn}", "w+") as output_review_file:
            for review in review_jsons:
                output_review_file.write(json.dumps(review) + "\n")
            output_review_file.close()
