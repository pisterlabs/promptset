import json
import re
import time
import openai

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)

args = parser.parse_args()

openai.api_key = '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'

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
            time.sleep(60)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion

string_data = '["Stainless Steel Hex Nut, 1/4-20 (Pack of 50)", "Aluminum Round Spacer, 1/2" OD, 1/4" Length (Pack of 25)", "Brass Flat Washer, #8 Screw Size, 0.203" ID, 0.5" OD (Pack of 100)", "Zinc Plated Steel Phillips Drive Pan Head Machine Screw, #6-32, 1/2" Length (Pack of 50)", "Black Oxide Finish Steel Socket Head Cap Screw, 5/16"-18 Thread Size, 1" Length (Pack of 10)"]'

pattern = r'(?<![ \[])"(?![,\]])'

replaced_data = re.sub(pattern, "'", string_data)

print(replaced_data)

data = []
with open(args.input_file) as f:
    for line in f.readlines():
        data.append(json.loads(line))
print(len(data))

# count = 0
# examples = []
# for i, d in enumerate(data):
#     # print(d['pred'])
#     pred = d['pred']
#     pattern = r"\[.*?\]"
#     match = re.search(pattern, pred)
#     try:
#         example = json.loads(pred)
#     except json.JSONDecodeError:
#         pattern = r'(?<![ \[])"(?![,\]])'
#         pred = re.sub(pattern, "'", pred)
#         try:
#             example = json.loads(pred)
#         except json.JSONDecodeError:
#             print(pred)
#             count += 1
    #         prompt = "Product title: " + d['title'] + '\nPlease predict at least 5 other products titles. \n Format: ["title1", "title2", "title3", "title4", "title5"], do not say any word or explain. \n'
    #         pred = get_completion_with_retries(prompt)
    #     print(pred)
    #     example = json.loads(pred)
    # print(example)
    # examples.append(example)


count = 0
examples = []
for i, d in enumerate(data):
    # print(d['pred'])
    pred = d['pred']
    pattern = r"\[.*?\]"
    match = re.search(pattern, pred)
    try:
        example = json.loads(pred)
    except json.JSONDecodeError:
        # print(pred)
        # print()
        if pred.startswith('["'):
            if pred.endswith('" ]'):
                pred = pred.replace('" ]', '"]')
            elif pred.endswith('",]'):
                pred = pred.replace('",]', '"]')
            # cut tail
            idx = pred.rfind('"]')
            if idx != -1 and idx != len(pred) - 2:
                pred = pred[:idx+2]

            # replace quotes
            pattern = r'(?<![ \[])"(?![,\]])'
            pred = re.sub(pattern, "'", pred)
            # replace "xxx "a" yyy"
            pred = re.sub(r'(?<!,) "(.*?)"(?!,)', r" '\1' ", pred)

            # replace "xxx 1", 2"
            pred = re.sub(r'(", )(?!")', r"', ", pred)

            # replace /
            pred = pred.replace("\\", "\\\\")

            
        try:
            example = json.loads(pred)
        except json.JSONDecodeError:
            # count += 1
            pred_split = '[' + pred + ']'
            try:
                example = json.loads(pred_split)
            except json.JSONDecodeError:
                preds = pred.split('\n')
                preds = [x.strip() for x in preds]
                example = []
                for p in preds:
                    try:
                        e = json.loads(p)
                        example.append(e)
                    except json.JSONDecodeError:
                        example.append(p)
                    # print(p)
    #         prompt = "Product title: " + d['title'] + '\nPlease predict at least 5 other products titles. \n Format: ["title1", "title2", "title3", "title4", "title5"], do not say any word or explain. \n'
    #         pred = get_completion_with_retries(prompt)
    #     print(pred)
    #     example = json.loads(pred)
    # print(example)
    if len(example) < 2:
        count += 1
        # print(len(example))
        # print(d['uid'])
        # print(d['pred'])
        # print()
        # print(pred)
        # print(example)
    # if d['uid'] == 'B0007SXIMM':
    #     print(pred)
    #     x = json.loads(pred)
    examples.append(example)


with open(args.save_to, "w") as outfile:
    for i in range(len(data)):
        outfile.write(
            json.dumps(
                {
                    "id": data[i]['uid'],
                    "output": examples[i]
                }
            ) + "\n"
        )

print(count)

# examples = []
# with open(args.save_to) as f:
#     for line in f.readlines():
#         print(len(json.loads(line)['output']))
#         examples.append(json.loads(line))
