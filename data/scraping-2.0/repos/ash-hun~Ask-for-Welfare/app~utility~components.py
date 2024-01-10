import re
import utility
import math
import random
import streamlit as st
from openai import OpenAI

def make_content_card(FOLDER_PATH, PAGE_TITLE):
    md_contents = utility.read_markdown_files(FOLDER_PATH)

    for i in range(0, math.ceil(len(md_contents)/3)):
        globals()['c{}'.format(i*3)], globals()['c{}'.format(i*3+1)], globals()['c{}'.format(i*3+2)] = st.columns((1, 1, 1))

    for i in range(0,len(md_contents)):
        split_md_contents = md_contents[i].split(sep='\n', maxsplit=1)
        # 이걸 하나의 마크다운으로 바꾸면 카드 단위로 링크 가능.
        globals()['c{}'.format(i)].image("img/" + str(random.randint(1, 20)) + ".jpg", use_column_width=True) # 이미지 랜덤으로 삽입
        globals()['c{}'.format(i)].markdown(f'###### [{PAGE_TITLE}](https://www.bokjiro.go.kr/ssis-tbu/index.do) · 조회수 : {random.randint(200, 1000)}') # 제목은 앞부분 잘라서 / 조회수는 랜덤으로
        globals()['c{}'.format(i)].markdown('#### [' + re.sub( r"\n+|#", " ", split_md_contents[0] ) + '](https://www.bokjiro.go.kr/ssis-tbu/index.do)')
        globals()['c{}'.format(i)].write('\n')
        globals()['c{}'.format(i)].write('##### [' + re.sub( r"\n+|#", " ", split_md_contents[1][:50] ) + '...](https://www.bokjiro.go.kr/ssis-tbu/index.do)') # 한번만 스플릿해서 뒤에 20자.

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url("https://github.com/ash-hun/Ask-for-Welfare/blob/main/assets/lighttheme.png?raw=true");
                background-repeat: no-repeat;
                background-size: 90%;
                position: relative;

                margin-left: 10px;
                margin-top: 10px;

                padding-top: 100px;
                padding-left : 5px;
                padding-right : 5px;
                background-position: 10px 10px;
            }
            [data-testid="stSidebarNav"]::before {
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chat_input_value():
    client = OpenAI() 

    # js 가 여기 경로 아니면 저장 못하게함 ㅎㅎ;
    file = "/Users/siina/Downloads/stt_audio.mp3"
    res = utility.STT(file, client)
    # res = "탈북자를 위한 복지 제도"
    clean_text = res.replace("\n", "")
    
    return clean_text


def chat_output_value(txt):
    client = OpenAI() 
    # txt = """
    # 탈북자를 위한 복지정책은 다음과 같습니다.

    # 교육 지원: 북한 학력을 국내 학력으로 인정하고, 장학금, 멘토링, 대안교육시설, 방과후공부방, 그룹홈, 화상영어교육 및 학습지, 통일전담교육사 운영 등의 교육 서비스를 지원합니다. 통일부 및 남북하나재단에 문의 후 신청할 수 있습니다.
    # """
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=txt,
    )
    response.stream_to_file("output.mp3")

