import os
import random
from datetime import datetime, timedelta

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

openai.api_key = "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"

now_date = datetime.now()

# round to nearest 15 minutes
now_date = now_date.replace(minute=now_date.minute // 15 * 15, second=0, microsecond=0)

# split into date and time objects
now_time = now_date.time()
now_date = now_date.date() + timedelta(days=1)

def generate_prompt(destination="", arrival_to="", arrival_date="", arrival_time="", departure_from="", departure_date="", departure_time="", additional_information="", **kwargs):
    return f'''
Prepare trip schedule for {destination}, based on the following information:
* Arrival To: {arrival_to}
* Arrival Date: {arrival_date}
* Arrival Time: {arrival_time}
* Departure From: {departure_from}
* Departure Date: {departure_date}
* Departure Time: {departure_time}
* Additional Notes: {additional_information}
'''.strip()


def submit():    
    prompt = generate_prompt(**st.session_state)

    # generate output
    output = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.5,
        top_p=1,
        frequency_penalty=2,
        presence_penalty=0,
        max_tokens=1024
    )
    
    st.session_state['output'] = output['choices'][0]['text']

# Initialization
if 'output' not in st.session_state:
    st.session_state['output'] = '--'

st.title('GPT-3 맞춤형 여행 플래너')
st.subheader('당신의 상황에 맞게 여행 일정을 추천해주는 GPT-3 기반 맞춤형 여행 플래너입니다')

with st.form(key='trip_form'):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader('Destination')
        example_destinations = ['Paris', 'London', 'New York', 'Tokyo', 'Sydney', 'Hong Kong', 'Singapore', 'Seoul', 'Busan', 'Jeju']
        # 사용자가 도시를 직접 선택하도록 변경합니다.
        destination = st.selectbox('도시 선택', example_destinations)
        st.form_submit_button('Submit', on_click=submit)

    with c2:
        st.subheader('Departure')

        st.selectbox('Departure From', ('Airport', 'Train Station', 'Bus Station', 'Ferry Terminal', 'Port', 'Other'), key='departure_from')
        st.date_input('Departure Date', value=now_date + timedelta(days=1), key='departure_date')
        st.time_input('Departure Time', value=now_time, key='departure_time')

    with c3:
        st.subheader('Arrival')

        st.selectbox('Arrival To', ('Airport', 'Train Station', 'Bus Station', 'Ferry Terminal', 'Port', 'Other'), key='arrival_to')
        st.date_input('Arrival Date', value=now_date, key='arrival_date')
        st.time_input('Arrival Time', value=now_time, key='arrival_time')



st.text_area('Additional Information', height=200, value='I want to visit as many places as possible! (respect time)', key='additional_information')

st.subheader('Trip Schedule')
st.write(st.session_state.output)
