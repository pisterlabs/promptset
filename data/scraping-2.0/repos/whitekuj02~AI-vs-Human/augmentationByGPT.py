import random
import os
import openai

import pandas as pd
import numpy as np
import time

random.seed(int(time.time()))

text = []
label = []
## 기존 데이터 셋 ##

train_data = pd.read_csv('./datasets/train.csv')

text = []
label = []
for i in range(len(train_data)):
    R = train_data.iloc[i,1:5].astype(str)
    l = train_data.iloc[i,5]
    one_hot_label = [0,0,0,0]
    one_hot_label[l-1] = 1
    for d,a in zip(R,one_hot_label):
        text.append(d)
        label.append(a)

## 네이버 1000개 ##
# naver_train_data = pd.read_csv('./datasets/ratings_train.txt', sep="\t", encoding="utf-8")

# naver_train_data.drop("id", axis=1, inplace=True)
# naver_train_data.dropna(subset=["document"], inplace=True)
# naver_train_data["label"] = 0

# 랜덤하게 1000개의 인덱스를 선택
# sample_indices = random.sample(range(len(naver_train_data)), 1000)
# sample_indices = range(500)

# for idx in sample_indices:
#     text.append(naver_train_data["document"].iloc[idx])
#     label.append(naver_train_data["label"].iloc[idx])
## augmentation 

print("train data len :" + str(len(text)))

aug_num = 100
api_key = "..."# 오픈 소스 올릴 시 삭제 바람

openai.api_key = api_key
MODEL = "gpt-3.5-turbo"

text_aug=[]
label_aug=[]
for d,l in zip(text, label):
    # random_values = random.sample(range(len(text)), 1)

    USER_INPUT_MSG = f"""
        문장: {d} \
        과 비슷한 리뷰를 "리뷰: [생성된 문장]" 형식으로 1개만 작성해줘.
    """

    # 과 비슷한 리뷰를 "리뷰:" 형식으로 1개만 작성해줘.

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are maker that text augmentation"},
            {"role": "user", "content": USER_INPUT_MSG}, 
        ],
        temperature=0.5
    )

    print(response['choices'][0]['message']['content'])

    text_generate = response['choices'][0]['message']['content'][4:]
    text_aug.append(text_generate)
    label_aug.append(l)
    
    df = pd.DataFrame({
        "text" : [text_generate],
        "label" : [l]
    })

    # 파일이 존재하는지 확인
    write_header = not os.path.exists("./datasets/aug.csv")

    df.to_csv("./datasets/aug.csv", mode = "a", header=write_header, index=False)
    time.sleep(1)
    

# df = pd.DataFrame({
#     "text" : text_aug,
#     "label" : label_aug
# })

# df.to_csv("./datasets/augmentation_naver.csv", mode='w') # mode = "a", header=False
