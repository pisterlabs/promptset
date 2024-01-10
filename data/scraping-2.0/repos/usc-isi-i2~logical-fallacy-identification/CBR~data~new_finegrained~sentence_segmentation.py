import pandas as pd
import os
import openai
from tqdm import tqdm
import re
import time
import joblib

for split in ["train", 'dev', 'test']:

    source_data = pd.read_csv(f"wo_augmentation/{split}.csv")
    source_data_text_splitted_dict = dict(zip(
        source_data['text'].tolist(),
        source_data['splitted'].tolist()
    ))

    df = pd.read_csv(f"{split}.csv")
    all_texts = df['text'].tolist()

    # shuffle all_texts
    # random.shuffle(all_texts)

    openai.api_key = "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"
    splitted_sentences = []

    cnt = 0

    pbar = tqdm(total=len(all_texts), leave=False)

    while True:
        try:
            if all_texts[cnt] in source_data_text_splitted_dict:
                splitted_sentences.append(
                    source_data_text_splitted_dict[all_texts[cnt]])
                cnt += 1
                pbar.update(1)
                continue
            response = openai.Completion.create(
                model="code-davinci-002",
                prompt=f"Girl:\" I'm worried that my friend is mad at me.\" Friend: \"I wonder what you call a male lady bug?\"\nsplitted: \n1: Girl:\" I'm worried that my friend is mad at me.\" \n2: Friend: \"I wonder what you call a male lady bug?\"\n####\nIf the state can require car seats for small children and infants, they can just as easily require mothers to breast-feed instead of using formula.\nsplitted:\n1: If the state can require car seats for small children and infants, \n2: they can just as easily require mothers to breast-feed instead of using formula.\n####\nNot tipping your waiter is like stealing money right out of someone's wallet.\nsplitted: \n1: Not tipping your waiter is\n2: like stealing money right out of someone's wallet.\n####\nMr. Casal was very tired because he had no energy.\nsplitted:\n1: Mr. Casal was very tired because\n2: he had no energy.\n####\nYou should certainly be the one who washes the dishes -- you are just so good at it!\nsplitted:\n1: You should certainly be the one who washes the dishes --\n2: you are just so good at it!\n####\nAre you still a heavy drinker?\nsplitted:\n1: Are you still a heavy drinker?\n####\nIf we don’t teach teens to work harder, the human race is doomed.\nsplitted:\n1: If we don’t teach teens to work harder,\n2: the human race is doomed.\n####\nThis has to work so it will work.\nsplitted:\n1: This has to work so it will work.\n####\n{all_texts[cnt]}\nsplitted:\n",
                temperature=0,
                max_tokens=486,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["####"]
            )

            result = response['choices'][0]['text']

            splits = re.split(r'\n?\d:', result)
            splits = [split.strip() for split in splits if split.strip() != '']
            splitted_sentences.append('[SEP]'.join(splits))

            cnt += 1
            pbar.update(1)

            if cnt == len(all_texts):
                break

        except Exception as e:
            print(e)
            time.sleep(60)

    try:

        df['splitted'] = splitted_sentences
        df.to_csv(f"{split}.csv", index=False)
    except:
        joblib.dump(splitted_sentences, f"{split}.pkl")
