import openai
import streamlit as st

# OpenAI API 연결
openai.api_key = "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"

# OpenAI GPT-3 모델 설정
model_engine = "text-davinci-002"
prompt_template = "I am a personal fitness coach. Based on your current fitness level, goals, available equipment, and health status, I recommend the following workout plan: \n\nFitness level: {} \nGoals: {} \nEquipment available: {} \nHealth status: {} \n\nWorkout Plan:"

# Streamlit 앱 구성
st.title("맞춤형 운동 계획 생성기")
fitness_level = st.selectbox("현재 체력 수준", ["초보", "중급", "고급"])
goals = st.multiselect("목표", ["체중 감량", "근육 증가", "체력 향상"])
equipment = st.multiselect("사용 가능한 장비", ["아이언 웨이트", "덤벨", "바벨", "스쿼트 랙"])
health_status = st.selectbox("건강 상태", ["정상", "부상 중", "질병"])

if st.button("운동 계획 생성"):
    prompt = prompt_template.format(fitness_level, goals, equipment, health_status)
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    workout_plan = response.choices[0].text
    st.success(workout_plan)
