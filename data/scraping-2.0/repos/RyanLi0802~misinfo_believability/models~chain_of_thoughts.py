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
    user_prompt = f'{row["text"]}\n\nThe potentially misleading tweet above is tweeted by someone with {row["user_followers_count"]} followers and has {row["retweet_count"]} retweets.\n\nIf this tweet were spread to a population that\n - is highly polarized\n - is susceptible to misinformation\n - is susceptible to bandwagon cues and gets easily influenced by popular figures\n - doesn\'t think rationally or critically\n - gets easily influenced by emotion, especially negative ones\n - tends to believe information that aligns with their prior beliefs\nwould its message be believed by many or by few? Think in terms of how the tweet might appeal to the population and consider the tweet\'s metadata, sentiment, and topic. Do NOT consider the tweet\'s own factual accuracy, as a factually inaccurate tweet can still appear believable to the general public.'
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": gpt_prompts.system_cot},
                    {"role": "user", "content": gpt_prompts.user1_cot},
                    {"role": "assistant", "content": gpt_prompts.assistant1_cot},
                    {"role": "user", "content": gpt_prompts.user2_cot},
                    {"role": "assistant", "content": gpt_prompts.assistant2_cot},
                    {"role": "user", "content": gpt_prompts.user3_cot},
                    {"role": "assistant", "content": gpt_prompts.assistant3_cot},
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