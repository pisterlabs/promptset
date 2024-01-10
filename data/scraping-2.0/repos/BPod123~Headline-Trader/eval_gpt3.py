import openai
from dotenv import load_dotenv
import os
from data import load_data_from_pickle
import pandas as pd
load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
def get_examples_from_training(index_name, random=True):
    pass
def get_prediction_zeroshot(headlines, index_name="S&P 500"):
    examples = get_examples_from_training()
    prompt = \
        f"Given the following headline data, say whether the {index_name} went up or down after one week of their release.\n" \
        f"Headlines: Ukrainian president Zelensky is pleading for more stern weaponry. CNN\'s Ben Wedeman goes to a drill where Ukrainian troops are trying out the fresh US-supplied rifles. How To Reduce Your Mortgage Payment Recalculate Your House Payment Now After months of military buildup and brinkmanship, Russia launched an unprecedented assault on Ukraine in late February. Can You Refinance With $0 Out Of Pocket? Germany records first competitive victory against Italy, Hungary thrashes England in UEFA Nations League Woman says she was forced to give up daughter to alleged rapist -- and pay child support Pakistanis told to drink less tea as nation grapples with economic crisis Los Angeles DA George Gascon recall group says it has collected required signatures to put matter on ballot NATO Fast Facts\n" \
        f"Market: down\n" \
        f"Headlines: {headlines}\n" \
        f"Market:"
    response = openai.Completion.create(
        engine="text-curie-001",
        prompt=prompt,
        temperature=0.9,
        max_tokens=4,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )
    prediction = response["choices"][0]["text"]
    prediction = prediction.strip().lower().replace("'", "")
    if prediction not in ["up", "down"]:
        print(prediction)
        prediction = 0
    elif prediction == "up":
        prediction = 1
    else:
        prediction = 0
    return prediction

def eval(ds):
    gt = list()
    pred = list()
    correct = 0
    for news, outcome in ds:
        outcome = 1 if outcome.Label else -1
        headlines_all = " ".join(news.title)
        for i in range(min(len(headlines_all) // 500, 5)):
            headlines = headlines_all[i*500 : (i+1)*500]
            pred.append(get_prediction_zeroshot(headlines))
            gt.append(outcome)
            if pred[-1] == gt[-1]:
                correct += 1
    print("true num up labels", len([1 for i in gt if i == 1]))
    print("predicted up labels", len([1 for i in pred if i == 1]))
    print("num samples", len(gt))
    print(correct / len(gt))
    return correct / len(gt)


if __name__ == '__main__':
    ds = load_data_from_pickle("Headlines.db")
    train_ds = ds[0]
    val_ds = ds[1]
    test_ds = ds[2]
    print(eval(train_ds[:10]))