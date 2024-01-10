# 라이브러리 호출 파트
import pandas as pd
import numpy as np
import openai
from tqdm import tqdm
import json
import csv

#################################입력해야 실행 가능####################################
openai.api_key = "<API KEY>"
csv_file_path = '<loaninfo_data_path>'
######################################################################################

# 데이터 호출
info_df = pd.read_csv(csv_file_path, encoding='cp949')

# df에 결측치가 포함되었을때 True를 반환
def isnan(number,column):
    is_nan = pd.isna(info_df.loc[number, column])
    return is_nan

# 필요항목 추출
result_tray={}
for index in range(len(info_df)):
    a,b,c,d,e,f,g,h,i = "","","","","","","","",""
    if isnan(index, '대출상품내용') ==False:
        a='대출상품내용: '+info_df.loc[index]['대출상품내용']
    if isnan(index, '최대금액') ==False:
        b='최대금액: '+str(info_df.loc[index]['최대금액'])
    if isnan(index, '대상') ==False:
        c='대상: '+info_df.loc[index]['대상']
    if isnan(index, '상품설명') ==False:
        d= '상품설명: '+info_df.loc[index]['상품설명']
    if isnan(index, '대상상세') ==False:
        e= '대상상세: '+str(info_df.loc[index]['대상상세'])
    if isnan(index, '금리') ==False:
        f= '금리: '+str(info_df.loc[index]['금리'])
    if isnan(index, '대출한도') ==False:
        g= '대출한도: '+str(info_df.loc[index]['대출한도'])
    if isnan(index, '상환방법') ==False:
        h= '상환방법: '+str(info_df.loc[index]['상환방법'])
    if isnan(index, '상환방법상세') ==False:
        i= '상환방법상세: '+str(info_df.loc[index]['상환방법상세'])
    result = ""
    result += a + "\n" if a else ""
    result += b + "\n" if b else ""
    result += c + "\n" if c else ""
    result += d + "\n" if d else ""
    result += e + "\n" if e else ""
    result += f + "\n" if f else ""
    result += g + "\n" if g else ""
    result += h + "\n" if h else ""
    result += i + "\n" if i else ""
    result_tray[index]=result
    
# Chat GPT를 사용한 프롬포트 생성
for index in tqdm(range(len(result_tray))):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "주어지는 대출 상품을 이용할 것 같은 사람들의 특징을 5개의 예시를 들어줘.앞뒤 불필요한 말 붙이지 말고 답만 내놔."},
                {"role": "user", "content": result_tray[index]}
                ]
            )
        result = completion.choices[0].message
        json_file_path = "C:\\Users\\mhkim\\Desktop\\coding\\samples.json"

        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)  # JSON 파일 내용 읽기
        data[index] = result["content"]
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    except:
        print(f"{index} error")
        pass
    


