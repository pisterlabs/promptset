import json
import re
from argparse import ArgumentParser
import time
import openai

parser = ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--gt_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--dataset', choices=['amazon', 'wiki', 'eurlex'])

args = parser.parse_args()

input_string = "[3] > [4] > [9] > [10] > [1] > [2] > [5] > [6] > [7] > [8]"

# Extract the numbers from the input string
numbers = [int(num.strip("[] ")) for num in input_string.split(">")]

print(numbers)

openai.api_key = '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'
# openai.api_key = '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'

def get_completion_with_retries(prompt, return_text=True, reduce_length=False, tqdm=None):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30
            )
            break
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                print('reduce_length')
                return 'ERROR::reduce_length'
            # self.key_id = (self.key_id + 1) % len(self.key)
            # openai.api_key = self.key[self.key_id]
            time.sleep(10)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion

data = []
with open(args.input_file) as f:
    for line in f.readlines():
        data.append(json.loads(line))
print(len(data))

preds = []
with open(args.gt_file) as f:
    for line in f.readlines():
        preds.append(json.loads(line))

# count = 0
# for i, d in enumerate(data):
#     pred_num = d['pred']
#     cands = preds[i]['output']
#     if any(char.isalpha() for char in pred_num):
#         count += 1
#         d['pred'] = cands
#     else:
#         numbers = [int(num.strip("[] ")) for num in pred_num.split(">")]
#         ordered_cands = []
#         for i in numbers:
#             if i <= len(cands):
#                 ordered_cands.append(cands[i-1])
#         d['pred'] = ordered_cands

count = 0
for i, d in enumerate(data):
    pred_num = d['pred']
    cands = preds[i]['output']
    # print(pred_num)
    pattern = r"\[.*?\]"
    match = re.search(pattern, pred_num)
    if match:
        list_format = match.group()
    else:
        # print(pred_num)
        # continue
        list_format = "[" + pred_num + "]"
    try:
        pred_num = json.loads(list_format)
    except json.JSONDecodeError:
        count += 1
        cands_prompt = ''
        for cid, cand in enumerate(cands):
            cand_prompt = '[' + str(cid + 1) + '] ' + cand + '\n'
            cands_prompt += cand_prompt
        if args.dataset == 'amazon':
            prompt_select = "**Task**: Given a query product, select the top 10 most relevant products from a list of candidates.\n**Query product title**: " \
            + d['title'] + "\n**Candidates**:\n" + cands_prompt + "\n**Output format**: A list of integers representing the indices of the top 10 most relevant products. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation. "
        elif args.dataset == 'wiki':
            prompt_select = "**Task**: From the following candidate list of Wikipedia pages, select top 10 that would be most relevant for the 'See also' section of the given page:\n**wiki title**: " \
            + d['title'] + "\n**Candidates**:\n" + cands_prompt + "\n**Output format**: A list of integers representing the indices of the top 10 most possible titles. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation."
        elif args.dataset == 'eurlex':
            prompt_select = "**Task**: From the following candidate list of labels, select top 10 that would be most relevant for the EU legislative document:\n**doc title**: " \
            + d['title'] + "\n**Candidates**:\n" + cands_prompt + "\n**Output format**: A list of integers representing the indices of the top 10 most possible titles. Example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] \nOnly ouput the list, do not include any description or explanation."

        
        prompt = prompt_select

        # print(prompt)
        # print(len(prompt), len(cands))
        prompt = prompt[:13000]
        # print(prompt)
        result = get_completion_with_retries(prompt)
        # print(result + '\n')
        pattern = r"\[.*?\]"
        match = re.search(pattern, result)
        if match:
            list_format = match.group()
        else:
            # print(pred_num)
            # continue
            list_format = "[" + result + "]"
        try:
            pred_num = json.loads(list_format)
        except json.JSONDecodeError:
            print(result)
            pred_num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(pred_num)
    print(len(cands))
    ordered_cands = [cands[i-1] for i in pred_num if i <= len(cands)]
    d['pred'] = ordered_cands
print(count)

with open(args.save_to, "w") as outfile:
    for d in data:
        outfile.write(
            json.dumps(
                {
                    "id": d['uid'],
                    "output": d['pred']
                }
            ) + "\n"
        )

