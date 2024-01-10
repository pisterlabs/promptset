import pandas as pd
import os
import openai
from tqdm import tqdm
import re
import time
import joblib

for split in ["train", 'dev', 'test']:

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
            print(result)
            splitted_sentences.append(result)
            cnt += 1
            pbar.update(1)

            if cnt == len(all_texts):
                break

        except Exception as e:
            print(e)
            time.sleep(60)

    try:

        # splitted_sentences = df['splitted'].tolist()

        results = []

        for splitted_sentence in splitted_sentences:
            splits = re.split(r'\n?\d:', splitted_sentence)
            splits = [split.strip() for split in splits if split.strip() != '']
            results.append('[SEP]'.join(splits))

        df['splitted'] = results
        df.to_csv(f"{split}.csv", index=False)
    except:
        joblib.dump(splitted_sentences, f"{split}.pkl")
