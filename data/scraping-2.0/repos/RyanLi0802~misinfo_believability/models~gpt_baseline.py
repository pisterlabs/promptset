import pandas as pd
import openai
import data_utils
import gpt_prompts

openai.api_key = open('.openai.key').read().replace('\n', '').replace('\r', '').strip()

few, many = data_utils.get_test_set()

test_data = pd.concat([few, many], axis = 0)
targets = [0] * 50 + [1] * 50
labels = []
# print(test_data)
# print(len(targets))

for idx, row in test_data.iterrows():
    user_prompt = f'{row["text"]}\n\nIs the potentially misleading tweet above believable by many or believable by few?'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": gpt_prompts.system_baseline},
            {"role": "user", "content": gpt_prompts.user1_baseline},
            {"role": "assistant", "content": gpt_prompts.assistant1_baseline},
            {"role": "user", "content": gpt_prompts.user2_baseline},
            {"role": "assistant", "content": gpt_prompts.assistant2_baseline},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    res = response.choices[0]['message']['content'].strip().replace('.', '').split(" ")[-1]
    label = 1 if res == 'many' else 0
    labels.append(label)
    print(response.choices[0]['message']['content'])
    print(res)
    
data_utils.eval_metrics(labels, targets)