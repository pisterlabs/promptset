import pandas as pd
import sys
import os

import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# sys.path.append("../config")
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

# Module
# from .Answer_Prompt_Template import chatchain
from .Answer_Prompt_Template import Answer_Template
from .Rule_Validation.Rule_Validation import checking_answer_template
from .Make_Json import Make_Issues, Make_Paragraph

# Async
import time
import asyncio
from langchain.llms import OpenAI

# 이제 async_generate는 인덱스와 함께 결과를 반환
async def async_generate(llm, policy, instruction, index):
    resp = await llm.agenerate([Answer_Template(policy, instruction)])
    return index, resp.generations[0][0].text

# 결과를 인덱스와 함께 수집하고, 이를 다시 정렬
# 각 기재항목에 맞는 하이퍼파라미터와 모델로 질의
async def generate_concurrently(df):
    for i in range(len(df)):
        match (df['part'][i]):
            case "제목 및 서문":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 처리 목적":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 처리 및 보유 기간":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "처리하는 개인정보의 항목":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "만 14세 미만 아동의 개인정보 처리에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 제3자 제공에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보 처리업무의 위탁에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 국외 이전에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 파기 절차 및 방법에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "미이용자의 개인정보 파기 등에 관한 조치":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "정보주체와 법정대리인의 권리·의무 및 행사방법에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 안전성 확보조치에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보를 자동으로 수집하는 장치의 설치·운영 및 그 거부에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "행태정보의 수집·이용·제공 및 거부 등에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "추가적인 이용·제공 관련 판단 기준":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "가명정보 처리에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보 보호책임자에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "국내대리인 지정에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보의 열람청구를 접수·처리하는 부서":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "정보주체의 권익침해에 대한 구제방법":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "영상정보처리기기 운영·관리에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

            case "개인정보 처리방침의 변경에 관한 사항":
                llm = OpenAI(temperature=0.05, top_p=0.3, model_name='gpt-4-1106-preview')

        task = async_generate(llm, df['matched_part'][i], df['instruction'][i], i)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    # 인덱스를 기준으로 결과를 정렬합니다.
    sorted_results = sorted(results, key=lambda x: x[0])
    return [result[1] for result in sorted_results]

# 최종 답변 주는 함수
# 1. 최종 답변 주는 LLM질의
# 2. 백엔드로 넘길 JSON 데이터의 중요한 부분 반환
def Answer_Model(df, text, issue_id_num, original_df):

    # 최종 위반 결과(법률위반, 법률위반가능, 지침위반, 권장사항)
    process_Law_Violate = 0
    process_Law_Danger = 0
    process_Guide_Violate = 0

    # 계속 위반한 규칙들이 들어갈 리스트
    process_Issues = []

    ans = ""
    answer = asyncio.run(generate_concurrently(df))
    print("비동기로 전부 뽑은 값", answer)
    ## 비동기로 뽑은 answer들은 먼저 끝내는대로 들어감, 순서가 일정하지 않음, 다시 순서 맞춰줘야함

    original_indexes = original_df.index.tolist()
    # 원본 데이터프레임의 인덱스

    for i in range(len(df)):

        original_index = original_indexes[i]
        # 최종 질의 프롬프트가 LLM에 Input으로 들어감 -> answer에서 결과 나옴
        #answer = chatchain.run(policy=df['matched_part'][i], instruction=df['instruction'][i])

        # ans는 그냥 결과 보여주기 위함
        ans += str(i+1)+">" +" " + "instruction: " + df['part'][i] + "에 해당하는 가이드라인은 이렇습니다." + "\n" + answer[i] + "\n\n\n\n"

        # Make_Issues 함수를 통해 Json형식에 들어갈 데이터 추출
        # issue_json은 리스트 형태로 된 각 규칙으 딕셔너리 집합체 [{}, {}, {}]
        issue_json, process_Law_Violate_temp, process_Law_Danger_temp, process_Guide_Violate_temp, issue_id_num = Make_Issues(answer[i], i, text, df, original_index, issue_id_num)

        process_Law_Violate += process_Law_Violate_temp
        process_Law_Danger += process_Law_Danger_temp
        process_Guide_Violate += process_Guide_Violate_temp

        # 계속 리스트에 추가해감
        process_Issues += issue_json

    process_Paragraph = Make_Paragraph(df, text)

    print("Make_Issues의 최종결과입니다.", process_Issues)
    print("Make_Paragraph의 최종결과입니다.", process_Paragraph)

    return ans, process_Paragraph,process_Issues, process_Law_Violate, process_Law_Danger, process_Guide_Violate