import pandas as pd
import openai
import data_utils
import gpt_prompts

openai.api_key = open('.openai.key').read().replace('\n', '').replace('\r', '').strip()

few, many = data_utils.get_test_set()

test_data = pd.concat([few, many], axis = 0)
targets = [0] * few.shape[0] + [1] * many.shape[0]
labels = []
# print(test_data)
# print(len(targets))

for idx, row in test_data.iterrows():
    user_prompt = f'A user with {row["user_followers_count"]} followers has tweeted the following potentially misleading message:\n\n"{row["text"]}"\n\nThe tweet has received {row["retweet_count"]} retweets.\n\n{gpt_prompts.user_template_cot2}'
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": gpt_prompts.system_cot2},
                    {"role": "user", "content": gpt_prompts.user1_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant1_cot2},
                    {"role": "user", "content": gpt_prompts.user2_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant2_cot2},
                    {"role": "user", "content": gpt_prompts.user3_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant3_cot2},
                    {"role": "user", "content": gpt_prompts.user4_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant4_cot2},
                    {"role": "user", "content": gpt_prompts.user5_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant5_cot2},
                    {"role": "user", "content": gpt_prompts.user6_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant6_cot2},
                    {"role": "user", "content": gpt_prompts.user7_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant7_cot2},
                    {"role": "user", "content": gpt_prompts.user8_cot2},
                    {"role": "assistant", "content": gpt_prompts.assistant8_cot2},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            res = response.choices[0]['message']['content'].strip().replace('.', '').split(" ")[-1]
            label = 1 if res == 'many' else 0
            labels.append(label)
            print(idx)
            print(user_prompt)
            print()
            print(response.choices[0]['message']['content'])
            print()
            break
        except Exception as e:
            continue
    
data_utils.eval_metrics(labels, targets)