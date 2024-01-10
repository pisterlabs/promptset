# 라이브러리 불러오기
import streamlit as st
import openai
from PIL import Image
import requests
from bs4 import BeautifulSoup
import random
import folium
import datetime
from streamlit_folium import folium_static
import base64

# API 인증
openai.api_key = "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"

# 탭 생성
tabs = ["Home","streamlit", "ChatGPT Text 대화하기", "ChatGPT Image 생성하기", "AI 기반 추천 시스템", "AI 기반 제안서 작성", "AI 기반 자기소개서 작성", "공공데이터 API 연동", "공공데이터 API 연동-음식 추천"]

# 사이드바 생성
st.sidebar.title("Navigation")
page = st.sidebar.radio("", tabs)

# Home
if page == "Home":
    st.title("ChatGPT API를 활용한 웹서비스 개발")

# Streamlit
elif page == "Streamlit":
    st.title("Streamlit 실습")
    st.subheader("작성 예정")

# ChatGPT text 대화하기
elif page == "ChatGPT Text 대화하기":
    st.title('ChatGPT Text 대화하기')
    st.write("자연어 처리 기술을 기반으로 한 ChatGPT가 제공하는 놀라운 대화 경험을 즐겨보세요!")

    # 채팅을 위한 text를 입력 받는다.
    with st.form(key='my_form'):
        content = st.text_area(
            "ChatGPT에게 요청할 내용을 입력하세요", 
            height=300,
            help="예시: '한국에서 가장 많이 먹는 음식은 무엇인가요?'"
        )

        # 제출 버튼을 추가한다.
        submitted = st.form_submit_button(label='ChatGPT 요청')

    # ChatGPT API 요청 처리
    if submitted:
        with st.spinner('ChatGPT의 응답을 기다리는 중입니다...'):
            completion = openai.Completion.create(
                engine="davinci",
                prompt=content,
                max_tokens=1024, 
                n=1,
                stop=None,
                temperature=2,
            )
            message = completion.choices[0].text.strip()
        
        # 결과 출력
        st.write(f'ChatGPT 응답:')
        st.write(message)

# ChatGPT Image 생성하기
elif page == "ChatGPT Image 생성하기":
    st.title('ChatGPT Image 생성하기')
    st.write("OpenAI의 GPT 모델을 활용하여 단어나 문장으로 원하는 이미지를 생성합니다!")

    # 채팅을 위한 text를 입력 받는다.
    prompt = st.text_input(
        label="chatGPT에게 그리고 싶은 그림을 입력하세요",
        help="예시: 'A picture of a cat playing with a ball'"
    )

    # chatGPT를 연동해본다.
    if st.button(label='chatGPT 요청', key='chatGPTButtonImage'):
        response = openai.Image.create(
            prompt=prompt,
            n=4,  # 4개의 이미지 생성
            size="512x512"  # 이미지 크기 512x512로 설정
        )

        # 생성된 이미지 URL들을 리스트로 가져옴
        image_urls = [data['url'] for data in response['data']]

        # 2행 2열로 4개의 이미지를 출력
        col1, col2 = st.columns(2) # 2개의 컬럼 생성
        with col1:
            st.image(image_urls[0], caption=f'Generated Image 1', use_column_width=True)
            st.image(image_urls[1], caption=f'Generated Image 2', use_column_width=True)
        with col2:
            st.image(image_urls[2], caption=f'Generated Image 3', use_column_width=True)
            st.image(image_urls[3], caption=f'Generated Image 4', use_column_width=True)


# AI 기반 추천 시스템(text + Image 생성)
elif page == "AI 기반 추천 시스템":
    # Streamlit 앱 제목 설정
    st.title("AI 기반 추천 시스템")
    st.write("OpenAI의 ChatGPT를 이용하여 자연어 처리를 기반으로 한 인공지능 대화와 대화에 맞는 이미지 생성을 제공합니다.")

    model_engine = "text-davinci-002"
    model_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."

    # DALL-E 이미지 생성 API 함수 정의
    def generate_image(prompt):
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        return response['data'][0]['url']

    # ChatGPT 대화 모델과 이미지 생성 API를 활용한 추천 시스템 함수 정의
    def recommendation_system(user_input):
        # ChatGPT API를 이용하여 대화 생성
        chat_response = openai.Completion.create(
            engine=model_engine,
            prompt=model_prompt + user_input,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
            frequency_penalty=0,
            presence_penalty=0
        )
        chat_message = chat_response.choices[0].text.strip()
        
        # OpenAI의 Image.create() 메서드를 이용하여 이미지 생성
        image_url = openai.Image.create(
            prompt=user_input,
            n=1,
            size="512x512"
        )['data'][0]['url']

        # 생성된 대화와 이미지를 반환
        return chat_message, image_url

    # Streamlit 앱 내용 설정
    user_input = st.text_input("사용자: ", "")

    if user_input:
        bot_response, image_url = recommendation_system(user_input)
        st.write("AI 챗봇: " + bot_response)
        st.image(image_url)


# AI 기반 제안서 작성 서비스text 생성 응용)
elif page == "AI 기반 제안서 작성":
    # Streamlit 앱 제목 설정
    st.title("AI 기반 제안서 작성")
    st.write("OpenAI의 ChatGPT를 이용하여 자연어 처리를 기반으로 한 제안서 작성을 도와줍니다.")

    # Streamlit 앱
    st.write("원하는 정보를 입력하고 제안서를 작성하세요.")

    # 제목, 회사명, 제안서의 내용 등 필요한 정보를 입력받는 창을 만듭니다.
    project_title = st.text_input("프로젝트 제목을 입력하세요.")
    company_name = st.text_input("회사명을 입력하세요.")
    project_summary = st.text_area("프로젝트 요약을 입력하세요.", height=200)
    project_description = st.text_area("프로젝트 설명을 입력하세요.", height=200)

    # ChatGPT에게 제안서 작성 요청
    if st.button("ChatGPT 제안서 작성 요청"):
        with st.spinner("제안서 작성 중입니다. 잠시만 기다려주세요..."):
            prompt = (f"제목: {project_title}\n"
                    f"회사명: {company_name}\n"
                    f"요약: {project_summary}\n"
                    f"설명: {project_description}\n"
                    "제안서 작성을 시작합니다.\n\n"
                    "다음은 우리 회사가 이 프로젝트를 수행하기에 완벽하게 적합한 이유입니다:\n\n")
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.7,
            )
            proposal = response.choices[0].text

        # 제안서 내용을 출력합니다.
        st.subheader("제안서 내용")
        proposal_lines = proposal.split("\n")
        for line in proposal_lines:
            st.write(line)

        # 제안서 작성 결과를 개괄식으로 보여주기 위해 출력하는 내용을 저장합니다.
        result = f"프로젝트 제목: {project_title}\n"\
                f"회사명: {company_name}\n"\
                f"프로젝트 요약: {project_summary}\n"\
                f"프로젝트 설명: {project_description}\n"\
                "제안서 내용:\n\n"\
                f"{proposal}"

        # 개괄적인 제안서 작성 결과를 출력합니다.
        st.subheader("제안서 작성 결과 개괄")
        st.write(result)

        def create_download_link(string, title="Download Text File", filename="file.txt"):
            """Create a download link for a given string."""
            b64 = base64.b64encode(string.encode()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{title}</a>'
            return href

        # 제안서 결과를 파일로 다운로드 받을 수 있는 버튼을 추가합니다.
        file_name = f"{project_title} 제안서.txt"
        file_download_link = create_download_link(result, file_name)
        st.markdown(file_download_link, unsafe_allow_html=True)

# AI 기반 자기소개서 작성 서비스(text 생성 응용)
elif page == "AI 기반 자기소개서 작성":
    # Streamlit 앱 제목 설정
    st.title("AI 기반 자기소개서 작성")
    st.write("OpenAI의 ChatGPT를 이용하여 자연어 처리를 기반으로 한 자기소개서 작성을 도와줍니다.")   

    # Streamlit 앱
    st.write("원하는 정보를 입력하고 자기소개서를 작성하세요.")

    # 이름 입력 받기
    name = st.text_input("이름을 입력하세요.")
    
    # 직무 리스트 생성 및 입력 받기
    careers = ["개발자", "디자이너", "영업", "마케터", "경영지원"]
    position = st.selectbox("원하는 직무를 선택하세요.", careers)
    
    # 역량 입력 받기
    skills = st.text_input("자신이 보유한 역량을 입력하세요.")

    # 자기소개서 생성 버튼
    if st.button("자기소개서 작성하기"):
        prompt = (f"작성자: {name}\n직무: {position} 지원\n\n안녕하세요. {position}로 취업을 희망하는 {name}입니다. "
                f"저는 {position} 역할에 필요한 {skills} 역량을 보유하고 있습니다. "
                f"특히 {position}에 필요한 {skills} 능력을 활용하여 이전에 {position}으로 일한 경험이 있습니다. "
                "제가 쌓은 경험과 역량을 회사에서 발휘하여 함께 성장해 나가고 싶습니다. "
                "부족한 점이 있을 수 있지만, 적극적인 자세로 업무에 임하겠습니다. 감사합니다.")
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.9,
            max_tokens=1024,
            n = 2,
            stop=None,
            timeout=30,
        )
        cover_letter = response.choices[0].text.strip()
        st.write(cover_letter)


elif page == "공공데이터 API 연동":
    st.title('공공데이터 API 연동')
    st.write('API 연동하여 날씨 데이터 가져오기')

    def get_weather(lat, lng):
        # 공공 API 호출을 위한 변수 설정
        service_key = "KQiiox%2BS6qJOu%2Bp3Jln5Mda5K49tPCEutxIU6ieMdfhF6gpk3eW0rHFy4P8sp9pK3QkL4QhYYh6nauczDqw7OA%3D%3D"
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
        params = {
            "serviceKey": service_key,
            "base_date": datetime.datetime.now().strftime('%Y%m%d'),
            "base_time": datetime.datetime.now().strftime('%H%M'),
            "nx": int((lng - 126.0) / 0.01),
            "ny": int((lat - 37.0) / 0.01),
            "dataType": "JSON"
        }

        # requests 모듈을 이용하여 API 호출
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.error(f"API 호출에 실패하였습니다. 에러코드: {response.status_code}")
        else:
            # JSON 형식의 API 응답을 파싱하여 처리합니다.
            json_data = response.json()
            sky_value = None
            for item in json_data['response']['body']['items']['item']:
                if item['fcstTime'] == '1200' and item['category'] == 'SKY':
                    sky_value = item['fcstValue']
                    break
            if sky_value is None:
                st.error("해당 날짜/시간에 대한 데이터가 없습니다.")
            else:
                return get_sky_status_kor(int(sky_value))

    def get_sky_status_kor(sky_code):
        if sky_code == 1:
            return '맑음'
        elif sky_code == 3:
            return '구름많음'
        elif sky_code == 4:
            return '흐림'
        else:
            return '알 수 없음'

    # 사용자로부터 위도/경도를 입력 받습니다.
    lat = st.number_input("위도 입력", value=37.5665, format="%.6f")
    lng = st.number_input("경도 입력", value=126.9780, format="%.6f")

    # Folium을 사용하여 지도를 생성합니다.
    m = folium.Map(location=[lat, lng], zoom_start=13)

    # 생성된 지도에 마커를 추가합니다.
    folium.Marker(location=[lat, lng], popup='현재 위치', icon=folium.Icon(color='red')).add_to(m)

    # get_weather 함수를 사용하여 날씨 정보를 가져옵니다.
    weather = get_weather(lat, lng)
    if weather is not None:
        st.write(f'현재 날씨: {weather}')

    # 생성된 지도를 Streamlit으로 출력합니다.
    folium_static(m)

elif page == "공공데이터 API 연동-음식 추천":
    st.title('공공데이터 API 연동-음식 추천')
    st.write('API 연동하여 날씨 데이터 가져와서 날씨 기반 음식 추천합니다(텍스트와 이미지)')
