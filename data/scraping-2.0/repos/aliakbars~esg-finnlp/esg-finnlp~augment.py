import os
import json
from unicodedata import normalize
import pandas as pd
from pprint import pprint
import openai
from dotenv import load_dotenv
import random


def generate(opp1, opp2, risk1):
    openai.api_key = ""
    prompt = f"""
Each line in the following list between ``` contains a snippet taken from a news article and the respective ESG impact identification.
ESG impact is one of 'Risk' and 'Opportunity'.
```
Opportunity: {opp1}
Risk: {risk1}
Opportunity: {opp2}
Risk:
""",
    print(prompt)
    print()
    response = openai.Completion.create(
        model="text-davinci-003",
        max_tokens=2000,
        n=5,
        prompt=prompt,
        temperature=0.9
    )

    response_content = [r["text"] for r in response["choices"]]
    return response_content

if __name__=="__main__":
    en_df = pd.read_json("esg-finnlp/data/raw/ML-ESG-2_English_Train.json")
    en_df["news_content"] = en_df["news_content"].apply(lambda c: normalize("NFKD", c))
    risk_samples = []
    augmented_data = []
    for i in range(100):
        opp1, opp2 = en_df[en_df["impact_type"] == "Opportunity"].sample(2)["news_content"].values
        risk_sample = en_df[en_df["impact_type"] == "Risk"].sample(1)
        risk_index = risk_sample.index[0]
        while risk_index in risk_samples:
            risk_sample = en_df[en_df["impact_type"] == "Risk"].sample(1)
            risk_index = risk_sample.index[0]
        risk1 = risk_sample["news_content"].values[0]
        risk_samples.append(risk_index)
        print()
        generated = generate(opp1, opp2, risk1)
        augmented_data.append({
            'prompt': {
                'opp': [
                    opp1,
                    opp2
                ],
                'risk': {
                    'id': str(risk_index),
                    'content': risk1
                }
            },
            'results': generated
        })
    with open('esg-finnlp/data/augments/result_en100_2.json', 'w') as fp:
        json.dump(augmented_data, fp, indent=2)

    # English Augment
    # with open('esg-finnlp/data/augments/result_en100_2.json', 'r') as fp:
    #     augmented_data = json.load(fp)
    # results = []
    # for ad in augmented_data:
    #     results += ad['results']
    # random.seed(1337)
    # random.shuffle(results)
    # for i in range(5):
    #     current_results = results[i*100:(i+1)*100]
    #     df = pd.DataFrame()
    #     df['sentence'] = current_results
    #     df['label'] = 'Risk'
    #     df.to_json(f'esg-finnlp/data/splits/en/{i+1}/augment.json', orient='records')

    # French translated to English augment
    # Split to 5 CV
    df = pd.read_csv('esg-finnlp/data/translation/fr_en.csv')
    results = df[df['impact_type'] == 'Risk']['news_content'].tolist()
    random.seed(1337)
    random.shuffle(results)
    for i in range(5):
        current_results = results[i*72:(i+1)*72]
        df = pd.DataFrame()
        df['sentence'] = current_results
        df['label'] = 'Risk'
        df.to_json(f'esg-finnlp/data/splits/en/{i+1}/fr.json', orient='records')