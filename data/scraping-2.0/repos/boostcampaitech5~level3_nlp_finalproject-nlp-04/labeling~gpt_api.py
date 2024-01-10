import os
from tqdm import tqdm

import pandas as pd

import openai

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]

def get_label(company, news: str) -> str:
    key = os.environ['OPENAPI_KEY']
    openai.api_key = key

    prompt = f'''### 역할:
                너는 어느 증권사 재무팀에서 일하고 있는 애널리스트야. 너의 주 업무는 뉴스 기사를 보고 {company} 입장에서 해당 기사가 "긍정" 혹은 "부정"인지 라벨링 하는 것이야. 이때 아래의 규칙을 따라야해. 
                
                ### 기사:
                {news}
                
                ### 규칙:
                1. 주어진 기사에 대해 하나의 ["긍정", "부정"]만을 선택하여 라벨링한다. 
                2. 기사는 "### 기사:" 부분부터 시작된다. 
                3. 출력은 "### 출력"와 같이 json 형식으로 출력한다. 
                4. 라벨링은 기사 전문에 대해서 라벨링한다. 
                5. 중립은 선택지에 없다. 
                6. 출력은 형식외에 아무것도 작성하지 않는다.

                ### 출력
                {{"label": "긍정", "reason": "새로운 투자 유치로 인해 해당 기업의 전망이 밝을것으로 예측된다. "}}
                '''

    response = get_completion(prompt)
    return response
    # return "### 출력{\"label\": \"긍정\", \"reason\": \"새로운 투자 유치로 인해 해당 기업의 전망이 밝을것으로 예측된다. \"}"

if __name__ == "__main__":
    path_file = ""
    pos_start = 0
    len_slice = 500

    df_news = pd.read_csv(path_file)[pos_start:pos_start+len_slice]
    df_news = df_news.drop_duplicates(["title"])
    df_news = df_news.dropna()

    list_result = []

    for idx, row in tqdm(df_news.iterrows(), total=df_news.shape[0]):
        try:
            list_result.append(get_label(row["company"], row["content_corpus"]))
        except Exception as e:
            print(f"idx: {idx}, err: {e}")
            list_result.append(f"idx: {idx}, err: {e}")
    
    df_news["label"] = list_result

    df_news.to_csv(f"./label_{pos_start}_to_{pos_start + len_slice}.csv")
    