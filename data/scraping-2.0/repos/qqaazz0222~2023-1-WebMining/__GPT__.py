# 주제 분류 모델
# 대주제 21개
subject = """A.농업, 임업 및 어업
B.광 업
C.제 조 업
D.전기, 가스, 증기 및 공기 조절 공급업
E.수도, 하수 및 폐기물 처리, 원료 재생업
F.건 설 업
G.도매 및 소매업
H.운수 및 창고업
I.숙박 및 음식점업
J.정보통신업
K.금융 및 보험업
L.부동산업
M.전문, 과학 및 기술 서비스업
N.사업시설 관리, 사업 지원 및 임대 서비스업
O.공공 행정, 국방 및 사회보장 행정
P.교육 서비스업
Q.보건업 및 사회복지 서비스업
R.예술, 스포츠 및 여가관련 서비스업
S.협회 및 단체, 수리 및 기타 개인 서비스업
T.가구 내 고용활동 및 달리 분류되지 않은 자가 소비 생산활동
U.국제 및 외국기관"""

# # 데이터 Classification
# subjectClassification = subject.split("\n")
# subjectClassification = [x.split(".") for x in subjectClassification]


# -*- coding: utf-8 -*-
import os
import openai


openai.api_key = "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"


messages = [{"role": "system", "content": subject}]


def classification():
    result = ["도구", "활용", "중등", "교사", "영어", "출제", "연수", "사례", "연구", "를", "중심"]

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    chatbot = completion.choices[0].message["content"].strip()


result = ["도구", "활용", "중등", "교사", "영어", "출제", "연수", "사례", "연구", "를", "중심"]
new = ", ".join(result)
print(new)
