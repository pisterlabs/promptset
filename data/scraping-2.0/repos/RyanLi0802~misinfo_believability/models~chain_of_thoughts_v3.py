import pandas as pd
import numpy as np
import openai
import data_utils
import gpt_prompts

openai.api_key = open('.openai.key').read().replace('\n', '').replace('\r', '').strip()

few, many = data_utils.get_test_set()
np.random.seed(seed=2)

# TODO: generate image description for all data samples
# find the intersection between our samples and Xinyi's samples with image descriptions
ids = pd.read_csv('../multimodal-data-xinyi.csv', sep='\t')['tweetId']
# few = few.merge(ids, on="tweetId", how="inner")
many = many.merge(ids, on="tweetId", how="inner").sample(n=50)

test_data = pd.concat([few, many], axis = 0)
targets = [0] * few.shape[0] + [1] * many.shape[0]
labels = []
# print(test_data)
# print(len(targets))

print(gpt_prompts.user_template_cot3)
print()

for idx, row in test_data.iterrows():
    user_prompt = f'A user with {row["user_followers_count"]} followers has tweeted the following potentially misleading message:\n\n"{row["text"]}"\n\nThe tweet has received {row["retweet_count"]} retweets.\n\n{gpt_prompts.user_template_cot3}'
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": gpt_prompts.system_cot3},
                    {"role": "user", "content": gpt_prompts.user1_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant1_cot3},
                    {"role": "user", "content": gpt_prompts.user2_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant2_cot3},
                    {"role": "user", "content": gpt_prompts.user3_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant3_cot3},
                    {"role": "user", "content": gpt_prompts.user4_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant4_cot3},
                    {"role": "user", "content": gpt_prompts.user5_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant5_cot3},
                    {"role": "user", "content": gpt_prompts.user6_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant6_cot3},
                    {"role": "user", "content": gpt_prompts.user7_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant7_cot3},
                    {"role": "user", "content": gpt_prompts.user8_cot3},
                    {"role": "assistant", "content": gpt_prompts.assistant8_cot3},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            res = response.choices[0]['message']['content'].strip().replace('.', '').split(" ")[-1]
            label = 1 if res.lower() == 'yes' else 0
            labels.append(label)
            print(idx)
            print(f'A user with {row["user_followers_count"]} followers has tweeted the following potentially misleading message:\n\n"{row["text"]}"\n\nThe tweet has received {row["retweet_count"]} retweets.\n\n')
            print()
            print(response.choices[0]['message']['content'])
            print()
            break
        except Exception as e:
            continue
    
data_utils.eval_metrics(labels, targets)