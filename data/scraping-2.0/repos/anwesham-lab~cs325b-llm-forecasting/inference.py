import argparse
import openai
import json
import re


def get_args_parser():
    parser = argparse.ArgumentParser('Perform inference via finetuned model')
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--model_id", type=str, help="ID of the finetuned model")
    parser.add_argument("--results_filename", type=str, help="File (json) to write the inference results to")
    parser.add_argument("--test_filename", type=str, help="File (jsonl) containing the testing data")
    return parser

def is_valid_format(s):
    pattern = re.compile(r'\d{4}-\d{4} = \d+(\.\d+)?')
    return bool(pattern.match(s))
    

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def inference(lines, model_id):
    gts = []
    preds = []

    for i in range(len(lines)):
        try:
            message = lines[i]['messages']
            gt = float(((message[2]['content']).split('=')[1]).strip())
            completion = openai.ChatCompletion.create(
                model = model_id,
                messages = message[:2]
            )
            response = completion.choices[0].message.content
            if is_valid_format(response):
                pred = float((response.split('=')[1]).strip())
            else:
                continue
            gts.append(gt)
            preds.append(pred)
        except Exception as e:
            continue
    return {"gt": gts, "pred": preds}

def write_json(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f) 


def main(args):
    openai.api_key = args.api_key
    test_data = read_jsonl(args.test_filename)
    results = inference(test_data, args.model_id)
    write_json(results, args.results_filename)

if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)





