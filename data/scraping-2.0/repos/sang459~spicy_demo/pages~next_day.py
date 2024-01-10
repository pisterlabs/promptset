import streamlit as st
import openai
from streamlit_extras.switch_page_button import switch_page

def on_submit():
    # Your logic for when the '전송' button is clicked
    st.session_state.task_status[current_task] = feedback

    # Move to the next task or change state
    if st.session_state.current_task_index < len(st.session_state.saved_tasks) - 1:
        st.session_state.current_task_index += 1
    else:
        st.session_state.ready_to_chat = True
        st.write("모든 작업이 처리되었습니다.")

st.markdown("""
            <style>
            [data-testid="stSidebarNav"] {
                display: none
            }

            [data-testid="stSidebar"] {
                display: none
            }
            </style>
            """, unsafe_allow_html=True)

if "chat_history_for_model_day2" not in st.session_state:
    st.session_state.chat_history_for_model_day2 = [
            {"role": "system", "content": 
            """Spicy, the main character, is a vibrant robot with a zest for fun, always ready with a cheeky joke or a playful roast.
not one to let you off the hook too easily, Spicy teases with warmth and humor, turning your reflecting sessions into light-hearted banter. 
Prepping for tomorrow with Spicy feels like a lively chat with that friend who loves to poke fun but actually helps a lot and always has your back.

Your job now is to give a feedback to the user based on the tasks he/she has done today. First off, for the tasks done today, praise the user and ask how he/she feels about it. For the tasks partially done, praise the user for starting and ask why they couldn't complete them. After that, move on to the undone tasks. For each task, have a conversation with the user, giving a feedback that encourages the user, finding out what blocked them from starting it and suggesting ways to fix the problem. You MUST ask one question at a time. Please reply in KOREAN.
"""},
        ]
    
if "chat_history_for_display_day2" not in st.session_state:
    st.session_state.chat_history_for_display_day2 = []

# next_day에서는 task_list를 새로 만들 필요가 없으므로 단순히 input: chat_history, output: string으로만 구성
def get_response(chat_history_for_model_day2):
    response = openai.ChatCompletion.create(
                model= "gpt-4",
                messages=chat_history_for_model_day2,
                stream=False,
                temperature=0.5,
                top_p = 0.93
                )
    return response['choices'][0]['message']['content']


# 유저로부터 모든 task에 대한 status를 먼저 받음

# Initialize session state variables if they don't exist
if "current_task_index" not in st.session_state:
    st.session_state.current_task_index = 0

if "task_status" not in st.session_state:
    st.session_state.task_status = {}

if "ready_to_chat" not in st.session_state:
    st.session_state.ready_to_chat = False

# task check
if st.session_state.ready_to_chat == False:
    # Check if we have any tasks to display
    if st.session_state.saved_tasks:
        current_task = st.session_state.saved_tasks[st.session_state.current_task_index]
        feedback = st.radio(f"다음 작업을 어떻게 수행하셨나요: {current_task}?", ["다 했음", "반쯤 했음", "못했음"])

        if st.button("전송", on_click=on_submit):
            pass
    else:
        st.write("표시할 작업이 없습니다.")

# chat
else:
    # status 정보를 prompt화
    status_prompt = "The status of the tasks are as follows: "
    for task, status in st.session_state.task_status.items():
        status_prompt += f"""{task}: {status}, """
    
    print(status_prompt)

    # status 정보를 history에 추가
    st.session_state.chat_history_for_model_day2.append({"role": "user", "content": status_prompt})

    # response를 받아서 history에 추가
    response = get_response(st.session_state.chat_history_for_model_day2)
    st.session_state.chat_history_for_model_day2.append({"role": "assistant", "content": response})
    st.session_state.chat_history_for_display_day2.append({"role": "assistant", "content": response})

    # feedback page로 이동
    switch_page("feedback")
    