import argparse
import os
import json
import shortuuid
import logging
import requests
from tqdm import tqdm
import openai


from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)

logFormatter = logging.Formatter("%(asctime)s %(message)s")
logger = logging.getLogger()

logger_logfn = "log_getModelAnswer_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fileHandler = logging.FileHandler("{0}/{1}.log".format("./logs/", logger_logfn))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

oaikey = None

def import_json(file_path):
    file_path = os.path.expanduser(file_path)
    f = open(file_path)
    data = json.load(f)
    f.close()

    return data

def run_eval(model_id, model_file, question_file, answer_file):
    models_jsons = import_json(model_file)

    models = list(filter(lambda models_jsons: models_jsons['model_id'] == model_id, models_jsons))

    if(len(models)==0):
        logger.error(f"model {model_id} not found in {model_file}")
        quit()
    else:
        model=models[0]

    logger.info(f"Model: {model_id}")

    ques_jsons = import_json(question_file)
    ans_jsons = []

    for i in range(len(ques_jsons)):
        logger.info(f"Question {i+1}/{len(ques_jsons)}")
        answer=get_model_answer(model, ques_jsons[i])
        ans_jsons.append(answer)

    logger.info(f"Writing answers to file {os.path.expanduser(answer_file)}")

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

def askOpenAI(model, question):
    assert(oaikey is not None)

    openai.api_key = oaikey

    for _ in range(10):
        try:
            response = openai.ChatCompletion.create(
                model=model["model_name"],
                temperature=model["metadata"]["temperature"],
                max_tokens=model["metadata"]["max_tokens"],
                messages=[
                    {
                        "role": "user",
                        "content": question["text"],
                    },
                ],
            )
            answer = response["choices"][0]["message"]["content"]

            return answer
        except Exception as e:
            logger.error(f"Error while askOpenAI: {e}")
            time.sleep(5)

    return "error, tried 10 times" 

def askObabooga(model, question):
    try:
        server=model["params"]["oobabooga-server"]
        params=model["metadata"]
        prompt=model["params"]["prompt_template"].format(question=question["text"])

        URI=f"{server}/api/v1/generate"

        request = {
            'prompt': prompt
        }

        for p in params:
            request[p]=params[p]   

        response = requests.post(URI, json=request)

        if response.status_code == 200:
            raw_reply = response.json()['results'][0]['text']
            content = raw_reply
            logger.info(content)
            return content
        else:
            logger.error("Error in POST, Status code {response.status_code}")

    except Exception as e:
        logger.error(e)
        return "error"

def get_model_answer(model, question):
    if model["type"] == "OpenAI":
        answer=askOpenAI(model, question)
    elif model["type"] == "oobabooga-api":
        answer=askObabooga(model, question)
    else:
        logger.error(f"unknown model type {mt}".format(mt=model["type"]))
        quit()

    answer_json= {
        "question_id": question["question_id"],
        "text": answer,
        "answer_id": shortuuid.uuid(),
        "model_id": model["model_id"],
        "metadata": model["metadata"],
    }

    return answer_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-file", type=str, default="table/model.json")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="table/question.json")
    parser.add_argument("-k", "--openaikey", type=str)
    args = parser.parse_args()

    oaikey = args.openaikey

    if args.answer_file is None:
        answer_file = "table/answer/answer_" + args.model_id.replace(":","_") + ".jsonl"
    else:
        answer_file=args.answer_file

    run_eval(
        args.model_id,
        args.model_file,
        args.question_file,
        answer_file
    )
