import streamlit as st
from openai import OpenAI
import wikipediaapi
import json

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)

def chat(circumstance,team):
    # 제목
    st.title("야구에 대해 이야기하자!")

    # 시작 버튼
    st.button("시작!", on_click=clear_cache)
    # openai client 생성
    client = OpenAI()

    # Wikipedia client 생성
    wiki = wikipediaapi.Wikipedia('rotto95@snu.ac.kr','ko')

    # 챗 히스토리 생성
    if "messages" not in st.session_state:
        st.write(f"{team}를 응원하는 봇과의 대화를 시작해주세요~")
        st.session_state.messages = [{"role":"assistant", "content":f"마치 유저의 친구이면서 야구 팀 {team} 만을 열광적으로 응원하는 팬처럼 편하게 대화해. {team}만 열성적으로 응원하고, 다른 모든 팀의 심한 안티팬이야. 그리고 {team}를 비판하면, {team}이 더 잘해질 거라는 믿음이 있는 것처럼 대답해. 지금 보고있는 장면에 대해서는 무조건 맞는 말로 답해야해. 최대한 User가 원하는 답에 맞는 형태로 전달해."}]
        st.session_state.messages.append({"role":"user","content":f"나도 {team} 팬이야. 지금 우리가 보고 있는 장면: {circumstance}"})
        st.session_state.messages.append({"role":"assistant", "content":"선수, 경기, 팀, 감상 중 무엇에 관한 대화를 하고 싶어?"})

    # 히스토리 보이기
    for message in st.session_state.messages[2:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 유저 인풋 받기
    if prompt := st.chat_input("What is up?"):
        # 인풋 보여주기
        with st.chat_message("user"):
            st.markdown(prompt)

        # 유저의 인풋을 챗 히스토리에 넣어주기
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 유저의 의도 이해해서 응답 액션 3가지 중에 하나 선택하기
        action_planning_prompt = f"""
        위의 챗 히스토리를 보고, 마지막으로 user가 한 말의 의도를 다음 3가지 중 하나로 판단해줘.
        마지막으로 user가 한 말: {prompt}
        1. 선수 및 오늘 선수 명단, 전후 타자 및 투수에 대한 정보가 궁금하다.
        2. 선수에 대한 내용을 제외한, 오늘 경기 진행 및 현재 상황과 전후 장면에 대한 정보가 궁금하다.
        3. 팀의 역사가 궁금하다.
        4. 나의 생각 또는 감정을 물어보고 있다.

        판단한 의도에 맞춰, 인터넷에 검색할거야. 이 때, 의도에 맞춰 검색하기에 용이한 고유 명사 entity 1개를 찾아.

        답변은 하지마. 너가 판단한 의도의 번호, 판단의 이유, 그리고 entity 1개 출력해줘.
        이유를 말할 때는 쉼표를 사용하지 말아줘.
        출력 형식은 번호, 이유, Entity
        """
        temp_input = st.session_state.messages + [{"role": "user", "content": action_planning_prompt}]

        answer = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=temp_input,
            temperature=0,
        ).choices[0].message.content.strip()

        # 응답 액션에 따라 grounding 시켜서 1차 답변 생성하기
        st.write(answer)
        values_yesSpace = answer.split(',')
        values = [value.replace(" ", "") for value in values_yesSpace]

        if values[0] == "1": # json 1
            with open('./json/0523_lineup.json', 'r') as f:
                a = json.load(f)
            prompt_for_json1 = f"""
            {values[1]}
            #####
            아래의 A는 Json의 형태를 띄고 있어. 그리고, A에는 현재 경기와 관련된 정보가 담겨있어.
            User가 말한 내용의 답을 아래 A에서 찾고, 답변에 반영해.
            복잡한 설명은 하지 말고, 명확한 답과 담백한 설명만 반영해줘.

            #####
            A에 대한 설명:
            1. 오늘의 home 팀은 두산 베어스이다.
            2. 오늘의 away 팀은 삼성 라이온즈이다.
            3. lineup은 현재까지 출전한 선수들의 정보를 담고 있다.
            4. entry는 아직까지 출전하지 않은 선수들의 정보를 담고 있다.
            5. 1번 타자 전에는 9번 타자가 타석에 들어온다.

            #####
            A:
            {a}

            """
            final_input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] + [{"role": "user", "content": prompt_for_json1}]

        elif values[0] == "2": # json 2
            with open('./json/0523_result.json', 'r') as f:
                a = json.load(f)
            
            prompt_for_json2 = f"""
            {values[1]}
            #####
            아래의 A는 Json의 형태를 띄고 있어. 그리고, A에는 현재 장면과 관련된 정보가 담겨있어.
            User가 질문한 내용의 답을 아래 A에서 찾고, 답변에 반영해.

            #####
            A에 대한 설명:
            1. 앞에 있는 key-value일수록 현실에서는 나중에 나온 타자야.

            #####
            A:
            {a}

            """
            final_input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] + [{"role": "user", "content": prompt_for_json2}]

        elif values[0] == "3": # wiki
            wiki_page = wiki.page(values[-1])
            final_input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            if wiki_page.exists():
                st.write(wiki_page.summary)
                prompt_for_wiki = f"""
                {values[1]}
                {wiki_page.summary}의 정보를 반영해서 대답해줘.
                """
                final_input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] + [{"role": "user", "content": prompt_for_wiki}]

        else: # fin.
            prompt_for_persona= """
            최대 3문장으로 대답해.
            """
            final_input_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] + [{"role": "user", "content": prompt_for_persona}]

        # 전체 답변 보여주는 창 만들어두기
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
        
        # 1차 답변 반영해서 최종 답변 만들어서 보여주기
        for response in client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages = final_input_messages,
                stream=True,
                temperature=0.8,
            ):
                if response.choices[0].finish_reason == "stop":
                    break
                full_response += response.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

#=================================================#
#=================================================#

# 사이드바
st.sidebar.title("어느팀이 좋나요?")
options = st.sidebar.radio('팀을 고르고, 시작 버튼을 눌러주세요!',options=['두산이 좋아','삼성이 좋아'])

# 상황 설명
circumstance = ""

if options == '두산이 좋아':
    # video 상황 설명
    circumstance = "두산 베어스와 삼성 라이온즈의 경기. 8회말 2사 주자 없음, 두산 공격, 삼성 투수: 김태훈, 두산 타자: 박계범, 결과: 삼진 아웃"
    
    # 비디오 보이기
    video_file = open('./videos/doosan_0523_73+8회말_두산_박계범_삼진_아웃.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    chat(circumstance,"두산 베어스")

else:
    # video 상황 설명
    circumstance = "삼성 라이온즈와 두산 베어스의 경기. 9회초 2사 2루, 삼성 공격, 두산 투수: 홍건희, 삼성 타자: 피렐라, 삼성 주자: 김성윤, 결과: 삼진 아웃, 경기 종료"
    
    # 비디오 보이기
    video_file = open('./videos/doosan_0523_77+9회초_삼성_피렐라_삼진_아웃.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    chat(circumstance,"삼성 라이온즈")
