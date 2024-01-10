import openai
import pandas as pd
from tqdm import tqdm
import os

openai.api_key = "PUT_YOUR_KEY"

templates = {
    1: 'problem description: {} Code snippet:{} Generate question title:',
    2: '',
    3: ''
}

lans = ['Ruby', 'Go']


def main():
    df = pd.read_csv('./data/Ruby_test.csv').astype(str)

    prediction_ls = []
    for idx, row in df.iterrows():
        inputDesc = row['desc']
        inputCode = row['code'][:8000]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": templates[1].format(inputDesc, inputCode)}
            ]
        )

        prediction = response['choices'][0]['message']['content']
        print(prediction)
        prediction_ls.append(prediction)
        with open('./result/ruby.pred.csv', 'a', encoding='utf-8') as f:
            f.write(prediction.replace('\n', ' ') + '\n')


if __name__ == '__main__':
    main()
