import openai
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.express as px
from io import BytesIO
from PIL import Image
import requests
from PIL import Image, ImageTk

# OpenAI API 인증 정보 설정
openai.api_key = "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"

# OpenAI API 모델 설정
model_engine = "davinci"

# GPT-3에 전달할 prompt 템플릿
prompt_template = """
당신의 체중은 {}kg, 키는 {}cm, 나이는 {}세이며, 활동 수준은 '{}'입니다. 
식이 제한 사항은 {}이고, 체중 감량 목표는 {}입니다. 
당신의 BMI는 {}입니다. 이는 {}에 해당합니다.

다음은 당신에게 맞는 맞춤형 다이어트 계획입니다.

---

"""

# BMI 계산 함수
def calculate_bmi(weight: float, height: float) -> float:
    return weight / ((height / 100) ** 2)

# BMI 범주 계산 함수
def calculate_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "저체중"
    elif bmi < 23:
        return "정상 체중"
    elif bmi < 25:
        return "과체중"
    elif bmi < 30:
        return "경도비만"
    elif bmi < 35:
        return "중등도비만"
    else:
        return "고도비만"
    
# Streamlit 앱 구성
st.title("맞춤형 다이어트 계획 생성기")
st.write("입력된 정보를 바탕으로 개인의 몸 상태, 건강 상태, 식습관 등을 고려하여 맞춤형 다이어트 계획을 제공합니다")
weight = st.number_input("현재 체중(kg)", min_value=40, max_value=300, step=1)
height = st.number_input("키(cm)", min_value=120, max_value=300, step=1)
age = st.number_input("나이", min_value=10, max_value=120, step=1)
activity_level = st.selectbox("활동 수준", ["최소", "보통", "높음"])
dietary_restrictions = st.multiselect("식이 제한 사항", ["없음", "채식주의자", "유당 민감성", "과민성 대장 증후군"])
weight_loss_goals = st.multiselect("체중 감량 목표", ["1kg 이하", "1-2kg", "2-3kg", "3-4kg", "4kg 이상"])

diet_plan = ""  # diet_plan 변수 초기화

if st.button("다이어트 계획 생성"):
    if weight is None or height is None or age is None or not dietary_restrictions or not weight_loss_goals:
        st.error("입력값이 부족합니다. 모든 값을 입력해주세요.")
    else:
        bmi = calculate_bmi(weight, height)
        bmi_category = calculate_bmi_category(bmi)
        prompt = prompt_template.format(weight, height, age, activity_level, dietary_restrictions, weight_loss_goals, round(bmi, 1), bmi_category)
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.6,
        )
        diet_plan = response.choices[0].text
        diet_plan = diet_plan.strip()  # 중복된 결과 제거
        st.success(diet_plan)
    
        # 생성된 다이어트 계획 출력
        st.subheader("나만의 맞춤형 다이어트 계획")
        st.write(diet_plan)

else:
    # 비만도 지수 계산 및 출력
    if weight is None or height is None:
        st.write("나의 체질량지수는 없습니다.")
    else:
        bmi = calculate_bmi(weight, height)
        bmi_category = calculate_bmi_category(bmi)
        st.subheader("나의 비만도 지수")
        st.write(f"나의 체질량지수는 {bmi:.2f} {bmi_category} 입니다.")
