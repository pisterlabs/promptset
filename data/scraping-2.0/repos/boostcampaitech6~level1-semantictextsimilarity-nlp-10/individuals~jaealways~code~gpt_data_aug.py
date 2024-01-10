import openai
import pandas as pd
import json
from tqdm import tqdm


openai.api_key = ""

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "augmentation"}
    ],
    temperature=0,
    max_tokens=256,
)


"""
1. Synonym Replacement (SR): Randomly choose n words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random.
2. Random Insertion (RI): Find a random synonym of a random word in the sentence that is not a stop word. Insert that synonym into a random position in the sentence. Do this n times.
3. Random Swap (RS): Randomly choose two words in the sentence and swap their positions.
Do this n times.
4. Random Deletion (RD): Randomly remove each word in the sentence with probability p.
"""


def refine(title: str, temperature=0, max_tokens=256):
    prompt = f"""
You are a generator of following Korean Sentences. Generate new full sentence with following condition.

Randomly remove each word in the sentence with probability p.

example
Input: 미세먼지 해결이 가장 시급한 문제입니다!
Output: 미세먼지 해결이 가장 문제입니다!


Input: {title}"""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content


if __name__ == '__main__':
    train_data = pd.read_csv('../data/train.csv')
    sentence1 = train_data['sentence_1'][train_data['binary-label']==1.0].tolist()

    # sentence1 = [
    #     "입사후 처음 대면으로 만나 반가웠습니다.",
    #     "주택청약조건 변경해주세요.",
    #     "앗 제가 접근권한이 없다고 뜹니다;;",
    #     "스릴도있고 반전도 있고 여느 한국영화 쓰레기들하고는 차원이 다르네요~",
    #     "미세먼지 해결이 가장 시급한 문제입니다!"
    # ]

    sen_aug={}

    for sen in tqdm(sentence1):
        try:
            tmp=refine(sen).split(': ')[1]
            sen_aug[sen]=tmp
        except:
            pass

    with open('train_GPTaug_bin1.json', 'w',encoding="UTF-8") as f:
        json.dump(sen_aug, f, indent=4, ensure_ascii=False)
