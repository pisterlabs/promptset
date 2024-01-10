import argparse, time
import openai
import json
import tqdm
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_set', type=str, default='test')
    parser.add_argument('--openai_api_key', default=None, type=str, required=True, help="API key to use GPT-3.")
    parser.add_argument('--preprocess', action='store_true', help="preprocess the ToMi dataset")
    parser.add_argument('--predict', action='store_true', help="Requests openai to generate responses")
    args = parser.parse_args()
    opt = vars(args)
    openai.api_key = args.openai_api_key
    input_path = f"ToMi/data/{opt['dataset_set']}.txt"
    if opt['preprocess']:
        preprocess_tomi(input_path, opt)
    continue_index = 0
    if opt['predict']:
        predict(continue_index, opt)
    gold, predictions = average_accuracy(opt)
    joint_accuracy(gold, predictions)


def joint_accuracy(gold, predictions):
    joint_results = []
    gold_joint = []
    predictions_joint = []
    for i, item in enumerate(predictions):
        if item == gold[i]:
            joint_results.append(True)
        else:
            joint_results.append(False)
        if (i + 1) % 6 == 0:
            prediction = all(element == joint_results[0] for element in joint_results)
            predictions_joint.append(prediction)
            gold_joint.append(True)
            joint_results = []
    accuracy_joint = accuracy_score(gold_joint, predictions_joint)
    print(f"GPT3 Accuracy: {accuracy_joint:.3f}")


def average_accuracy(opt):
    gold = []
    predictions = []
    with open(f'output/{opt["dataset_set"]}.txt') as f_in:
        for i, line in enumerate(tqdm.tqdm(f_in)):
            fields = json.loads(line)
            gold.append(str(fields['label']))
            predictions.append(fields['prediction'])
    accuracy = accuracy_score(gold, predictions)
    print(f"GPT3 Accuracy: {accuracy:.3f}")
    return gold, predictions


def predict(continue_index, opt):
    preprompt = get_preprompt()
    with open(f"output/{opt['dataset_set']}.txt", "a") as f_out:
        with open(f"ToMi_processed_data/{opt['dataset_set']}.txt") as f_in:
            for i, line in enumerate(tqdm.tqdm(f_in)):
                if i < continue_index:
                    continue
                print(i)
                fields = json.loads(line)
                prompt = preprompt + fields['context'] + "\n" + fields['question']
                fields['prediction'] = open_ai_finalanswer_request(prompt, i, 0).strip()
                fields['prompt'] = prompt
                f_out.write(json.dumps(fields) + "\n")

def get_preprompt():
    prompt = ""
    with open("ToMi_processed_data/prompt.txt") as f_in:
        for line in f_in:
            data = json.loads(line)
            prompt += data['context'] + '\n' + data['question'] + data['label'] + "\n\n"
    return prompt

def preprocess_tomi(input_path, opt):
    with open(f"ToMi_processed_data/{opt['dataset_set']}.txt", "a") as f_out:
        with open(input_path) as input_file:
            sample = dict()
            sample["context"] = ""
            for line in enumerate(input_file):
                print(line)
                if "?" in line[1]:
                    new_line = line[1].split("\t")
                    sample["context"] = sample["context"].strip()
                    sample["question"] = new_line[0].strip()
                    sample['label'] = new_line[1].strip() + " " + new_line[2].strip()
                    f_out.write(json.dumps(sample) + "\n")
                    sample = dict()
                    sample["context"] = ""
                else:
                    sample["context"] += line[1].strip() + "\n"


def open_ai_finalanswer_request(prompt, i, counter):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=30,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text'].strip()
    except:
        if counter < 3:
            time.sleep(10)
            return open_ai_finalanswer_request(prompt, i, counter + 1)
        else:
            print(prompt)
            print("continue from:" + str(i))
            exit()


if __name__ == "__main__":
    main()
