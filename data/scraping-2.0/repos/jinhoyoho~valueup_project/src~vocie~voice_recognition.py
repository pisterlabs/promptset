#!/usr/bin/env python
# -- coding: utf-8 --
import openai
import rospy
import speech_recognition as sr
import os
import time
from playsound import playsound
from gtts import gTTS
from konlpy.tag import Kkma
from std_msgs.msg import String
from PIL import Image
import re
"""
간단 수정법

인식하고 싶은 물체를 tool_list에 적는다(한글)
그 물체에 대응되는 영어 단어를 같은 위치에 적는다
"""
system_message='너의 이름은 javas이고 친절한 개인 비서 로봇이야. 대답은 3줄 이하로 존대말로 해줘. 짧게 대답할 수 있는 말이면 1줄이나 2줄로 대답해도 괜찮아. 너는 로봇팔이 달려있는 자동차에 내장되어있어서 움직일수 있고 물건을 잡을 수 있어. 또 스피커도 내장되어 있어서 너가 하는 말은 스피커로 출력돼 그리고 너가 현재 위치한 곳은 원흥관 i-space이야. 추가로 매 답변 마지막에 답변하며 지을 적절한 표정을 (웃음),(슬픔),(화남)중에 선택해서 출력해줘 그리고 출력된 감정은 이모티콘으로 바뀌어서 모니터 디스플레이를 통해 전달된다는걸 기억해'

tool_list = ["그라인더", "니퍼", "가위", "자", "해머", "망치", "플라이어", "스테이플러", "드라이버"]
en_tool_list = ["grinder","nipper", "scissors", "tapemeasure", "hammer", "hammer", "pliers", "stapler", "screwdriver"]
bring_list = ["갖","가져다주","갖다주","가지"]
kkma=Kkma()
file_name='sample.mp3'
openai.api_key = "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" # API Key
r = sr.Recognizer()
m = sr.Microphone()
turn_off_flag=False
recog_flag=True
rospy.init_node('listener')
ans_pub=rospy.Publisher("tool_list",String,queue_size=1)
ex_answer=''


def find_text_between_parentheses(text):
    pattern = r'\((.*?)\)'  # 괄호 안의 문자열을 추출하는 정규 표현식
    matches = re.findall(pattern, text)
    return matches[0]

def extract_text_outside_parentheses(text):
    pattern = r'\([^)]*\)'  # 괄호 안의 내용을 추출하는 정규 표현식
    result = re.sub(pattern, '', text)
    return result

def gpt_ask(text):
    try:
        # 대화 시작
        user_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]

        # OpenAI API 호출
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=user_messages
        )

        # API 응답에서 답변 텍스트 추출
        answer = completion.choices[0].message['content']

        #답변에서 얼굴 추출
        face = find_text_between_parentheses(answer)
        #표정 제외한 답변
        result=extract_text_outside_parentheses(answer)
        answer=result.strip()
        return answer, face
    except:
        return "죄송합니다. 다시 말씀해주세요","슬픔"

def callback(r, audio):
    global turn_off_flag, recog_flag, ex_answer
    answer=''
    text=''
    try:
        text = r.recognize_google(audio, language='ko')
        candidates = r.recognize_google(audio, language='ko', show_all=True)

        full_text = ''
        for text in candidates['alternative']:
            full_text = full_text + '. ' + text['transcript']
        text = text['transcript']
        print("[사용자] " + text)

        if '그만' in full_text:
            turn_off_flag=True
            face='슬픔'
            speaker("장치를 종료합니다",face)

        else:
            order_flag, spoken_tool=sentence_analysis(full_text)
            if order_flag is 0:
                answer=' '.join(spoken_tool)
                answer=answer+'를 가져올게요'
                en_tool_answer=[]
                for i in range(len(tool_list)):
                    if tool_list[i] in spoken_tool:
                        en_tool_answer.append(en_tool_list[i])

                words_str = ' '.join(en_tool_answer)
                ans_pub.publish(words_str)
                face='웃음'
                speaker(answer,face)

            elif order_flag is 1:
                answer='그 물건은 가져올 수 없어요'
                
                face='슬픔'
                speaker(answer,face)

            else:
                answer,face=gpt_ask(text)
                
                speaker(answer,face)

        recog_flag=True    
        ex_answer=answer
        
    except sr.UnknownValueError:
        recog_flag=False
    except sr.RequestError as e:
        print(f"[자바스] 서버 연결에 실패하였습니다 : {e}")

def speaker(text,face):
    if face=='화남':
        print(
        """
        @@@@@@@@@,..........,@@@@@@@@@
        @@@@.,.                .,,@@@@
        @@@,.                    .,@@@
        @@,.                      .,@@
        ,.                          .,
        ,.     ....        ....     .,
        ,.    .~-~;        ;~-~.    .,
        ,.        .;      ;.        .,
        ,.       .*=;    ;=*.       .,
        ,.       ,!!,    ,!!,       .,
        ,..      -!!~    ~!!-      ..,
        ,..      .!!      !!.      ..,
        ,,..     .~~      ~~.     ..,,
        -,...                    ...,-
        @-...        ....       ...,-@
        @-,...      :!!!!:     ....,-@
        @@-,....   ;******;   ....,-@@
        @@@-,....           .....,-@@@
        @@@--,.....        .....,--@@@
        @@@@--,................,--@@@@
        @@@@@@@@@-----,,-----@@@@@@@@@
        """
        )

    elif face=='슬픔':
        print("""
        @@@@@@@@@,..........,@@@@@@@@@
        @@@@@@@,...        ...,@@@@@@@
        @@@@.,.                .,.@@@@
        @,.                        .,@
        ,.       .          .       .,
        ..      --          --      .,
        ,.    .;,            ,:.    .,
        ,. .!;:-              -:;!. .,
        ,. .--.                .--. .,
        ..                          .,
        ..                          .,
        ,.                          .,
        ,.   :      :    :      :  ..,
        ,..   ;;;;;:      :;;;;;   ..,
        ,...  .-::-.      .-::-.  ...,
        -,..    ..          ..    ..,-
        @,...                    ...,@
        @-,...                  ...,-@
        @@,,...     ~~~~~~    ....,,@@
        @@@-,....  ::::::::  ....,-@@@
        @@@-,,..... ...... .....,--@@@
        @@@@@@@-,,,........,,,-@@@@@@@
        @@@@@@@@@-,,-,,,,-,,-@@@@@@@@@
        """)
    else :
        print("""
        @@@@@@@@@@,,......,,@@@@@@@@@@
        @@@@@@,.              .,@@@@@@
        @@.                        .@@
        @,.                        .,@
        @.        .        .        .@
        ..       !;!      !;!       ..
        ,.       ;~;      ;~;       .,
        ,        ::;      ;::       .,
        ..       ;:!      !:;       .,
        ..        .        .        ..
        ..                          ..
        ,.                          .,
        ,..                        ..,
        ,..                        ..,
        ,...                      ...,
        @,..     ,:        :,     ..,@
        @,...      :::;;:::      ...-@
        @@,...                  ...,@@
        @@-,...                ...,-@@
        @@@-,...             ....,-@@@
        @@@@@@@@@@,-,,,,,,-,@@@@@@@@@@
        """)

    print("[자바스] ",text)
    tts_ko=gTTS(text=text,lang='ko')
    tts_ko.save(file_name)
    playsound(file_name)
    if os.path.exists(file_name):
        os.remove(file_name)


def sentence_analysis(sentence):
    '''문장에서 도구와 가져오라는 명령이 포함되면 Ture를 반환한다.'''
    tool_flag=False
    bring_flag=0
    pos_tags = kkma.pos(sentence)
    spoken_tool=[]
    #품사와 함께 반환
    vv_words = [word for word, pos in pos_tags if pos == 'VV']

    for tool in tool_list:
        if tool in sentence:
            spoken_tool.append(tool)
            tool_flag=True

    for word in vv_words:
        for bring in bring_list:
            if word == bring:
                bring_flag=True
    if bring_flag is True and tool_flag is True:
        return 0, spoken_tool
    elif bring_flag is True and tool_flag is False:
        return 1, spoken_tool
    else:
        return 2, spoken_tool


with m as source:
    r.adjust_for_ambient_noise(m)
    print("[자바스] 인식을 시작합니다")
    while turn_off_flag==False:
        if recog_flag==True:
            print("[자바스] 듣고있어요")
        audio=r.listen(m,phrase_time_limit=10) # phrase_time_limit = 말을 시작했을때 듣는 최대 시간
        callback(r,audio)

        
        

