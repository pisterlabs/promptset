#!/usr/bin/python3
import base64
from openpibo.motion import Motion
import os
import sys
import time
import re
import random
import string
import speech_recognition as sr
import requests
import json
import openai
from threading import Thread

# openpibo module
import openpibo
from openpibo.device import Device
from openpibo.speech import Speech
from openpibo.audio import Audio
from openpibo.vision import Camera
from openpibo.oled import Oled

# path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))

from text_to_speech import TextToSpeech
from src.data import behavior_list
from src.NLP import NLP, Dictionary

import socket

localIP = "192.168.0.155"
localPort = 20001
bufferSize = 50000

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from uuid import uuid4
from firebase_admin import storage

from datetime import datetime

options = {
    '창세기': 'ge',
    '출애굽기': 'exo',
    '레위기': 'lev',
    '민수기': 'num',
    '신명기': 'deu',
    '여호수아': 'josh',
    '사사기': 'jdgs',
    '룻기': 'ruth',
    '사무엘상': '1sm',
    '사무엘하': '2sm',
    '열왕기상': '1ki',
    '열왕기하': '2ki',
    '역대상': '1chr',
    '역대하': '2chr',
    '에스라': 'ezra',
    '느헤미야': 'neh',
    '에스더': 'est',
    '욥기': 'job',
    '시편': 'psa',
    '잠언': 'prv',
    '전도서': 'eccl',
    '아가': 'ssol',
    '이사야': 'isa',
    '예레미야': 'jer',
    '예레미야 애가': 'lam',
    '에스겔': 'eze',
    '다니엘': 'dan',
    '호세아': 'hos',
    '요엘': 'joel',
    '아모스': 'amos',
    '오바댜': 'obad',
    '요나': 'jonah',
    '미가': 'mic',
    '나훔': 'nahum',
    '하박국': 'hab',
    '스바냐': 'zep',
    '학개': 'hag',
    '스가랴': 'zep',
    '말라기': 'mal',
    '마태복음': 'mat',
    '마가복음': 'mark',
    '누가복음': 'luke',
    '요한복음': 'john',
    '사도행전': 'acts',
    '로마서': 'rom',
    '고린도전서': '1cor',
    '고린도후서': '2cor',
    '갈라디아서': 'gal',
    '에베소서': 'eph',
    '빌립보서': 'phi',
    '골로새서': 'col',
    '데살로니가전서': '1th',
    '데살로니가후서': '2th',
    '디모데전서': '1tim',
    '디모데후서': '2tim',
    '디도서': 'titus',
    '빌레몬서': 'phmn',
    '히브리서': 'heb',
    '야고보서': 'jas',
    '베드로전서': '1pet',
    '베드로후서': '2pet',
    '요한1서': '1jn',
    '요한2서': '2jn',
    '3jn': '요한3서',
    '유다서': 'jude',
    '요한계시록': 'rev'
}

# 서비스 계정의 비공개 키 파일이름
cred = credentials.Certificate("/home/pi/AI_pibo2/src/play_scenario/CAP/pibo-aac5a-firebase-adminsdk-94r6k-faf630e478.json")
firebase_admin.initialize_app(cred, {
    #gs://pibo-aac5a.appspot.com
    'apiKey': "AIzaSyDF5r0N2u8KVFILABFMCtNbF7w4VtmQy54",
    'authDomain': "pibo-aac5a.firebaseapp.com",
    'databaseURL': "https://pibo-aac5a-default-rtdb.firebaseio.com",
    'projectId': "pibo-aac5a",
    'storageBucket': "pibo-aac5a.appspot.com",
    'messagingSenderId': "516104758418",
    'appId': "1:516104758418:web:81dd90f1624515f8f6f24b"
})
db = firestore.client()
# doc_ref.set({u'느낌' : 'BAD'})
bucket = storage.bucket() #기본 버킷 사용

NLP = NLP()
Dic = Dictionary()
tts = TextToSpeech()
device_obj = Device()
camera = Camera()
oled = Oled()

r = sr.Recognizer()
r.energy_threshold = 300
mic = sr.Microphone()


m = Motion()
m.set_profile("/home/pi/AI_pibo2/src/data/motion_db.json")

# biblefile = "/home/pi/AI_pibo2/src/data/bible.json"
# with open(biblefile, encoding='utf-8') as f:
#     bible = json.load(f)

client_id = "zq90hxu84o" # naver cloud platform - clova sentiment client id
client_secret = "B6jHEYSIrkCK4kTVbK8l1NXclQAUcnBu7bRXcEoo" # clova sentiment client password
url = "https://naveropenapi.apigw.ntruss.com/sentiment-analysis/v1/analyze" # naver sentiment url

# naver clova sentiment header
headers = {
    "X-NCP-APIGW-API-KEY-ID": client_id,
    "X-NCP-APIGW-API-KEY": client_secret,
    "Content-Type": "application/json" # jason 형식
}
openai.api_key = "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"

global text
global info
global verseinfo
global comment

now = datetime.now()
global today
today = now.strftime('%Y-%m-%d')


## 여기에 사용자 이름 넣기
global user_name
# user_name = '건호'

# 가장 최근 접속된 날짜 firebase 저장하기 위한 날짜형식
todayInKor = now.strftime('%m월 %d일')

# Generate text using the GPT model


# sad song list
sadsonglist = ["/home/pi/AI_pibo2/audio/위로노래/worry.mp3", "/home/pi/AI_pibo2/audio/위로노래/sadhalf.mp3",
                    "/home/pi/AI_pibo2/audio/위로노래/like.mp3", "/home/pi/AI_pibo2/audio/위로노래/correct.mp3", "/home/pi/AI_pibo2/audio/위로노래/hug.mp3",
                    "/home/pi/AI_pibo2/audio/위로노래/jina.mp3"]

# happy song list
happysonglist = ["/home/pi/AI_pibo2/audio/기쁜노래/boom.mp3", "/home/pi/AI_pibo2/audio/기쁜노래/candy.mp3", "/home/pi/AI_pibo2/audio/기쁜노래/step.mp3",
                    "/home/pi/AI_pibo2/audio/기쁜노래/cheer.mp3", "/home/pi/AI_pibo2/audio/기쁜노래/dream.mp3", "/home/pi/AI_pibo2/audio/기쁜노래/happ.mp3"]

# sososong list
sosonglist = ["/home/pi/AI_pibo2/audio/sososong/everything.mp3", "/home/pi/AI_pibo2/audio/sososong/kim.mp3", "/home/pi/AI_pibo2/audio/sososong/notme.mp3",
                "/home/pi/AI_pibo2/audio/sososong/nov.mp3", "/home/pi/AI_pibo2/audio/sososong/youth.mp3", "/home/pi/AI_pibo2/audio/sososong/live.mp3"]

def text_to_speech(text):
    filename = "tts.wav"
    print("\n" + text + "\n")
    # tts 파일 생성 (*break time: 문장 간 쉬는 시간)
    tts.tts_connection(text, filename)
    tts.play(filename, 'local', '-1500', False)     # tts 파일 재생

def text_to_speech2(text): # 원탁 아저씨
    filename = "tts.wav"
    print("\n" + text + "\n")
    # tts 파일 생성 (*break time: 문장 간 쉬는 시간)
    tts.tts_connection2(text, filename)
    tts.play(filename, 'local', '-1500', False)     # tts 파일 재생

def ends_with_jong(kstr):
    m = re.search("[가-힣]+", kstr)
    if m:
        k = m.group()[-1]
        return (ord(k) - ord("가")) % 28 > 0
    else:
        return

def lee(kstr):
    josa = "이" if ends_with_jong(kstr) else ""
    return josa

def aa(kstr):
    josa = "아" if ends_with_jong(kstr) else "야"
    return josa

def wait_for(item):
    while True:
        print(f"{item} 기다리는 중")
        break

###

def touching():

    touching = False

    _touch = ''
    for x in range(3):
        data = device_obj.send_cmd(Device.code_list['SYSTEM']).split(':')[1].split('-')
        _touch = data[1] if data[1] else "No signal"
        
        time.sleep(1)
        
        if _touch == 'touch':
            touching = True
        
    return touching

def songpick(songlist):
    ssongchoice = random.choice(songlist)
    return ssongchoice

def verse():
    doc_bible = db.collection(u'users').document(f'{user_name}').collection(u'성경').document(f'{today}')
    doc_bible.set({'말씀' : verseinfo, '주소' : info, '조언' : comment})

    oled.draw_image("/home/pi/AI_pibo2/src/data/icon/화면_default1.png")
    oled.show()
    text_to_speech(info)
    time.sleep(2)
    text_to_speech2(verseinfo)
    time.sleep(1)
    text_to_speech(comment)
    time.sleep(1)
    text_to_speech("오늘 말해줘서 고마워. 마지막으로 악수 하자!")
    m.set_motors([0,0,-70,-25,0,0,0,0,25,25])
    time.sleep(1)
    text_to_speech("남은 하루도 행복하기를 바랄게")
    time.sleep(4)
    m.set_motors([0,0,-70,-25,0,0,0,0,70,25])

def touch_scenario1():
    #oled.draw_image("/home/pi/AI_pibo2/src/data/icon/화면_로딩1.png")
    #oled.show()
    #tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav", out='local', volume=-1000, background=False)
    #behavior_list.neutral()
    text_to_speech(
        f"{user_name}{lee(user_name)}가 그런 일이 있었구나. 너를 안아주고 싶은데, 안아주지 못하니까 나의 이마를 쓰다듬어 줄래?")

    total = 0
    while True:
        time.sleep(1)
        data = device_obj.send_cmd(Device.code_list['SYSTEM']).split(':')[
            1].split('-')
        _touch = data[1] if data[1] else "No signal"
        print(_touch)
        if _touch == 'touch':
            total = total + 1
            print(total)
        if total == 1:
            device_obj.send_cmd(20, '255,255,123')
            oled.draw_image("/home/pi/AI_pibo2/src/data/icon/heart3.png")
            oled.show()
            text_to_speech("위로가 되는 거 같아! 더 쓰다듬어 줘!")
            
        elif total == 2:
            device_obj.send_cmd(20, '255,199,0')
            oled.draw_image("/home/pi/AI_pibo2/src/data/icon/heart1.png")
            oled.show()
            text_to_speech("내 위로를 너에게 가득 전달했어!")
            text_to_speech("너가 쓰다듬어주니까 나도 위로가 된다. 너한테도 위로의 마음이 전달되었으면 좋겠어.")
            break

def touch_scenario2():
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav",
             out='local', volume=-1000, background=False)
    text_to_speech(
        f"{user_name}{aa(user_name)}! 나는 항상 너를 응원해! 앞으로 우리는 더 좋은 일만 생길거야!")
    #m.set_speed(2, 25)
    #m.set_speed(8, 25)
    #m.set_motors([0, 0, 70, -25, 0, 0, 0, 0, -70, 25])
    text_to_speech("내 에너지를 전달해 줄게 내 이마를 쓰다듬어줘!")
    #m.set_motors([0, 0, -70, -25, 0, 0, 0, 0, 70, 25])

    total = -1
    while True:
        time.sleep(1)
        data = device_obj.send_cmd(Device.code_list['SYSTEM']).split(':')[1].split('-')
        _touch = data[1] if data[1] else "No signal"
        print(_touch)
        if _touch == 'touch':
            total = total + 1
            print(total)
        if total == 1:
            device_obj.send_cmd(20, '255,255,123')
            oled.draw_image("/home/pi/AI_pibo2/src/data/icon/heart3.png")
            oled.show()
            text_to_speech("위로가 되는 거 같아! 더 쓰다듬어 줘!")

        elif total == 3:
            device_obj.send_cmd(20, '255,199,0')
            oled.draw_image("/home/pi/AI_pibo2/src/data/icon/heart1.png")
            oled.show()
            text_to_speech("내 위로를 너에게 가득 전달했어!")
            text_to_speech("너가 쓰다듬어주니까 나도 위로가 된다. 너한테도 위로의 마음이 전달되었으면 좋겠어.")
            break

def heart_scenario():
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav", out='local', volume=-1000, background=False)
    #behavior_list.heart()

    text_to_speech(f"{user_name}{aa(user_name)}!! 너가 좋아하니 내가 너무 신나!! 내 심장소리 들려!?")
    tts.play(filename="/home/pi/AI_pibo2/audio/기타/심장박동.mp3", out='local', volume=-1000, background=False)
    text_to_speech("안 들린다면 내 가슴쪽을 봐줘!!")
    oled.set_font(size=50)

    for a in range(20):
        oled.draw_text((0, 0), str(a+150))  # (0,0)에 문자열 출력
        oled.show()  # 화면에 표시
        time.sleep(0.1)
        oled.clear()

    text_to_speech("내 심장이 너무 빨리 뛰어!! 진심으로 기뻐!")

    time.sleep(1)
    text_to_speech("너도 이만큼 기쁘다면 내 머리를 쓰다듬어줘.")
    
    if touching():
        oled.draw_image("/home/pi/AI_pibo2/src/data/icon/heart3.png")
        oled.show()
        text_to_speech("너도 그렇게 느꼈다니 너무 좋아!")   
    
    else:
        text_to_speech("별로 쓰다듬어주고 싶지는 않구나! 그럴 수 있지.")

def recording(expect, response):
    
    while True:
        with mic as source:
            print("say something\n")
            audio = r.listen(source, timeout=0, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio_data=audio, language="ko-KR")
            except sr.UnknownValueError:
                print("say again plz\n")
                continue
            except sr.RequestError:
                print("speech service down\n")
                continue

        # stt 결과 처리 (NLP.py 참고)
        answer = NLP.nlp_answer(user_said=text, dic=Dic)

        if answer == expect:
            text_to_speech(response)
            return 'y'

        elif answer == 'NO':
            return 'n'

        else:
            text_to_speech('잘 못 알아들었어. 다시 말해줄래?')
            recording(expect, response)
        
        break

def touch_test():
    print("touch test")
    total = 0
    for i in range(3):
        time.sleep(1)
        data = device_obj.send_cmd(Device.code_list['SYSTEM']).split(':')[1].split('-')
        _touch = data[1] if data[1] else "No signal"
        print(_touch)
        if _touch == 'touch':
            total = total + 1
    return total

def touch_scenario1():
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav",
             out='local', volume=-1000, background=False)
    
    #behavior_list.touch()

    text_to_speech(f"{user_name}{lee(user_name)}가 그런 일이 있었구나. 너를 안아주고 싶은데, 안아주지 못하니까 나의 이마를 쓰다듬어 줄래?")
    
    while True:
        touched = touch_test()
        if touched >= 1:
            device_obj.send_cmd(20, '0,0,255')
            text_to_speech("너가 쓰다듬어주니까 나도 위로가 된다. 너한테도 위로의 마음이 전달되었으면 좋겠어.")
            break

def sadSong():
    oled.draw_image("/home/pi/AI_pibo2/src/data/icon/화면_음표1.png")
    oled.show()
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav",
             out='local', volume=-1000, background=False)
    text_to_speech(f"{user_name}... 너를 위로해주고 싶어..! 노래를 불러주고 싶은데 괜찮을까? 노래를 듣고 싶다면 내 머리를 쓰다듬어줘")

    if touching():

        tts.play(filename=songpick(sadsonglist), out='local', volume=-2000, background=False)
        text_to_speech("위로가 되었으면 좋겠어..!")

    #answer = recording('YES',"노래 틀어줄게")

    else :
        text_to_speech("별로 노래가 듣고 싶지 않구나...알겠어...힘내")

def happySong():
    #behavior_list.praising()
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav", out='local', volume=-1000, background=False)
    
    text_to_speech(f"{user_name}{aa(user_name)}!! 너무 좋은 일이잖아!?!? 내가 신나는 노래 한곡 뽑아줄까? 괜찮으면 내 머리를 쓰다듬어줘.")

    if touching():
        tts.play(filename=songpick(happysonglist), out='local', volume=-2000, background=False)
        text_to_speech("기깔났다 정말! 너무 신나!")

    #answer = recording('YES',"노래 틀어줄게")

    else:
        text_to_speech("별로 노래가 듣고 싶지 않구나. 알겠어 노래는 틀지 않을게.")

def Cam():
    # Capture / Read file
    # 이미지 촬영
    img = camera.read()
    #img = cam.imread("/home/pi/test.jpg")
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/사진기소리.mp3",
             out='local', volume=-1000, background=False)

    todayTimeStamp = str(datetime.now().timestamp)
    camtodaynow = datetime.now()
    date_time = camtodaynow.strftime("%Y%m%d%H%M%S")
    
    jpgFile = f"{user_name}" + f"{date_time}" + ".jpg"
    
    camera.imwrite(jpgFile, img)
    img = camera.convert_img(img, 128, 64)
    camera.imwrite("smallpic.jpg", img)
    oled.draw_image("smallpic.jpg")
    oled.show()

    blob = bucket.blob(jpgFile)
    # blob = bucket.blob(f'{user_name}/'+file)
    #new token and metadata 설정
    new_token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token} #access token이 필요하다.
    blob.metadata = metadata

    #upload file
    blob.upload_from_filename(filename=jpgFile, content_type='image/jpeg')
    blob.make_public()
    print(blob.public_url)

    doc_pic = db.collection(u'users').document(f'{user_name}').collection(u'사진').document(f'{date_time}')
    tt = jpgFile.split(".jpg")
    finaltt = tt[0].split(f"{user_name}")
    doc_pic.set({'url' : blob.public_url, 'time' : finaltt[1]})

    os.system("rm %s" %jpgFile)

def takepic1():
    #behavior_list.do_photo()
    text_to_speech("너가 기분이 좋으니까 나도 기분이 좋다~! 웃는 모습을 담고 싶은데 우리 사진찍을래? 찍고 싶으면 내 머리를 쓰다듬어줘")
    
    #answer = recording('YES', "찍을게")

    if touching():
        text_to_speech("그래, 그럼 셋하고 찍을게. 하나, 둘, 셋!")
        Cam()
        text_to_speech(f"너무 보기 좋아 {user_name}{aa(user_name)}~. 내가 이따가 사진 보내줄게!")

    else :
        text_to_speech(f"알겠어! 사진은 찍지 않을게, {user_name}{aa(user_name)}~")

def takepic2():
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav", out='local', volume=-1000, background=False)
    #behavior_list.do_photo()
    
    text_to_speech("그렇구나, 기분이 너무 좋구나! 너가 지금 느끼는 것을 몸으로 표현해줘~")
    time.sleep(3)
    text_to_speech("지금 너무 행복해보인다! 내가 사진에 담아줄게! 그 자세로 있어주면 내가 사진을 찍어주려고 하는데 괜찮을까? 괜찮으면 내 머리를 쓰다듬어줘.")

    if touching():
        text_to_speech("그래, 그럼 셋하고 찍을게. 하나, 둘, 셋!")
        Cam()
        text_to_speech(f"너무 보기 좋아 {user_name}{aa(user_name)}~. 내가 이따가 사진 보내줄게!")
    
    else :
        text_to_speech(f"알겠어! 사진은 찍지 않을게, {user_name}{aa(user_name)}~")

def sosoSong():
    oled.draw_image("/home/pi/AI_pibo2/src/data/icon/화면_음표1.png")
    oled.show()
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav",
             out='local', volume=-1000, background=False)
    text_to_speech("그랬구나..! 아 맞아, 요즘 좋은 노래 많이 나오더라! 내가 최근에 좋아하는 노래 추천해주고싶은데 괜찮을까? 괜찮으면 내 머리를 쓰다듬어줘.")
    
    #answer = recording('YES',"내가 요즘 인디음악을 좋아해서 인디음악 들려줄게!")

    if touching():
        tts.play(filename=songpick(sosonglist), out='local', volume=-2000, background=False)
        text_to_speech("마음에 들었으면 좋겠다!")
    
    else:
        text_to_speech("별로 음악이 듣고 싶지 않구나! 그럴 수 있지")

def soso_takepic():
    oled.draw_image("/home/pi/AI_pibo2/src/data/icon/화면_카메라.png")
    oled.show()
    tts.play(filename="/home/pi/AI_pibo2/src/data/audio/물음표소리1.wav", out='local', volume=-1000, background=False)
    text_to_speech(f"그렇구나! {user_name}{aa(user_name)}!! 오늘 따라 옷 스타일이 멋진데, 사진 한장 찍어줄게! 괜찮아? 괜찮으면 내 머리를 쓰다듬어줘.")
    
    if touching():
        text_to_speech("그래 알겠어, 포즈 취해줘 찍어줄게! 하나, 둘, 셋!")
        Cam()
        text_to_speech(f"너무 보기 좋아 {user_name}{aa(user_name)}~. 내가 이따가 사진 보내줄게!")
    
    else :
        text_to_speech("알겠어, 사진은 찍지 않을게.")

def getName():
    text_to_speech("시작하기 전에 나와 연결을 먼저 해줘.")

    # 데이터그램 소켓을 생성
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    
    # 주소와 IP로 Bind
    UDPServerSocket.bind((localIP, localPort))

    print("UDP server up and listening")

    # 들어오는 데이터그램 Listen
    while (True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        address = bytesAddressPair[1]

        clientMsg = "Message from Client:{}".format(message)
        clientIP = "Client IP Address:{}".format(address)

        print(type(message))
        print(message)
        print(clientMsg)
        print(clientIP)

        message = message.decode('utf-8')
        message = base64.b64decode(message)
        message = message.decode('utf-8')
        print(message)

        if message is not None:
            text_to_speech(f"너의 이름은 {message}{lee(message)}구나! 만나서 반가워")
            return message

def Start():
    global text
    global user_name

    device_obj.send_cmd(20, '0,0,0') # 20 = eye, 0,0,0 = color rgb

    behavior_list.do_question_S()
    text_to_speech(f"안녕! 나는 은쪽이라고해!")

    #user_name = input("답변 (이름): ")
    user_name = getName()
    print(user_name)

    behavior_list.do_question_S()
    doc_feel = db.collection(u'users').document(f'{user_name}').collection(u'감정').document(f'{today}')
    doc_last = db.collection(u'users').document(f'{user_name}').collection(u'LAST').document('when')
    doc_diary = db.collection(u'users').document(f'{user_name}').collection(u'일기').document(f'{today}')

    doc_last.set({'last' : f'{todayInKor}'})

    text_to_speech(f"{user_name}{aa(user_name)} 오늘 하루 어땠어?")

    # TODO : change

    # while True:
    #     with mic as source:
    #         print("say something\n")
    #         audio = r.listen(source, timeout=0, phrase_time_limit=5)
    #         try:
    #             text = r.recognize_google(audio_data=audio, language="ko-KR")
    #         except sr.UnknownValueError:
    #             text_to_speech('잘, 못 알아들었어. 다시 말해줄래?')
    #             continue
    #         except sr.RequestError:
    #             print("speech service down\n")
    #             continue
    #     break
        
        # 감정분류 url (circulus)
        # url = "https://oe-napi.circul.us/v1/emotion?sentence="
        # url = url + text
        # response = requests.post(url)    global emotion

        # 감정분류 naver API
    
    text = input('사용자 감정 : ')

    doc_diary.set({'내용' : text})

    #오늘 기분이 어떤지?
    m = Thread(target=chatgpt, args=())
    o = Thread(target=activity, args=())

    m.daemon = True
    o.daemon = True

    m.start()
    o.start()
    m.join()
    o.join()
    #성경 구절 추천 여부

    behavior_list.do_question_S()
    text_to_speech("그러면 오늘의 성경구절 하나 추천해줄까?")

    # TODO : change
    # while True:
    #     with mic as source:
    #         print("say something\n")
    #         audio = r.listen(source, timeout=0, phrase_time_limit=5)
    #         try:
    #             text = r.recognize_google(audio_data=audio, language="ko-KR")
    #         except sr.UnknownValueError:
    #             text_to_speech('잘, 못 알아들었어. 다시 말해줄래?')
    #             continue
    #         except sr.RequestError:
    #             text_to_speech('음성인식 기능에서 뭔가 에러가 났어. 잠시만 기다려줘.')
    #             print("speech service down\n")
    #             continue
    text = input("오늘의 성경 구절 하나 추천해줄까 ( 그래 / 아니) : ")
    answer = NLP.nlp_answer(user_said=text, dic=Dic)
    
    if answer == 'YES':
        verse()
        
    
    elif answer == 'NO': 
        text_to_speech("오늘 말해줘서 고마워. 마지막으로 악수 하자!")
        m.set_motors([0, 0, -70, -25, 0, 0, 0, 0, 25, 25])
        time.sleep(1)
        text_to_speech("남은 하루도 행복하기를 바랄게")
        time.sleep(4)
        m.set_motors([0, 0, -70, -25, 0, 0, 0, 0, 70, 25])
        
        # else:
        #     text_to_speech('잘 못 알아들었어. 다시 말해줄래?')
        #     print("대답 기다리는 중")
        #     continue
        # break
        

def chatgpt():
    global text
    global info # 말씀
    global verseinfo # 주소
    global comment # 조언
    global user_name
    
    pre = "나에게 성경을 하나를 추천해 줘. 너는 대답을 이런 형식으로 해야해."
    rule1 = "1) 반말(~야)을 사용해줘. 친구에게 말을 하듯이 대답해줘."
    rule2 = "2) 대답을 시작할 때, 말씀의 제목을 먼저 말해줘. 예를 들어 '이사야 5장 5절이야'와 같은 형태로 대답을 시작해줘."
    rule3 = "3) 나를 부를 때는 '너' 라고 불러줘."
    rule4 = "4) 질문은 하지 마."
    rule5 = "5) 시편 같은 경우에는 장이 아니라 편이야 위에와 같은 형식으로 꼭 대답을 해."
    rule6 = "6) 대답을 할 때 존댓말을 절대 사용하지마."
    rule7 = "7) 말씀 주소를 먼저 말해줘. 그 다음에 한 문장으로 조언을 해줘."
    
    
    ex1 = "아래는 예시를 보여줄게. 같은 형식으로 대답을 해야해. "
    ex2 = "[(창세기 28장 15절)말씀이야. **너가 잃어버린 것이 얼마나 아까워서 불안하고 슬프겠지만, 하나님은 네가 가는 길에서 너를 지키시며, 네가 돌아가는 땅까지 너를 인도해주시리라 믿어봐.**] "
    ex3 = "위의 예시처럼 대답을 해줘. (성경 주소)말씀이야. ** 조언 **"
    
    rule8 = "8) 반드시 성경 한 절만 추천해줘야해."
    rule9 = "9) 조언을 반드시 '**'와 '**' 여기 사이 안에 넣어주어야해."
    rule10 = "10) 절대 조언에 성경구절 내용을 포함시켜서 말하지 말아줘. "
    rule11 = "11) 절대 존댓말을 사용하면 안돼. 대신 반말(~야)을 사용해줘. 친구에게 말하듯이. 그렇다고 명령조로는 말하지마"


    ex = ex1 + ex2 + ex3
    
    # Generate text using the GPT model
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        n=1,
        messages=[
            {"role": "system", "content": pre + rule1 + rule2 + rule3 + rule4 + rule5 + rule6 + ex + rule7 + ex + rule8 + rule9 + rule10 + rule11},
            {"role": "user", "content": pre + rule1 + rule2 + rule3 + rule4 + rule5 + rule6 + ex + rule7 + ex + text + rule8 + rule9 + rule10 + rule11},
        ])

    message = response.choices[0]['message']

    print("{}: {}".format(message['role'], message['content']))

    firstParse = message['content'].split(")")
    temp = firstParse[0]
    verseinfo = temp.replace('(', '')
    print(verseinfo)

    secondParse = firstParse[1].split("**")
    temp = secondParse[1]
    comment = temp.replace("**", "")

    ver = verseinfo.split()

    bible = ver[0]
    print(ver[0]) # 성경
    infochapt = ver[1]
    temp = ver[1].split('장') # 장
    chapter = temp[0]
    print(chapter)
    infoverse = ver[2] + '말씀이야'
    temp = ver[2].split('절') # 절
    verse = temp[0]
    print(verse)

    if verse.find('-') != -1:
        temp = verse.split('-')
        verse1 = temp[0]
        verse2 = temp[1]
    
    else:
        verse1 = verse
        verse2 = verse

    if bible in options:
        boutput = options[bible]
        print(boutput)
    else:
        print("입력된 값에 대한 옵션이 존재하지 않습니다.")

    url = "https://yesu.io/bible?lang=kor&doc="+ boutput + \
        "&start=" + chapter + ":" + verse1 + ":&end=" + chapter + ":" + verse2
    
    response = requests.get(url)
    print(response.text)
    print(response.json()[0].get('message'))

    info = bible + " " + infochapt + " " + infoverse
    verseinfo = response.json()[0].get('message')

def activity():
    global text

    doc_feel = db.collection(u'users').document(f'{user_name}').collection(u'감정').document(f'{today}')
    

    data = {
            "content": text
        }
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    rescode = response.status_code
    parse = response.json().get('document').get('sentiment')

    if (rescode == 200):
        print(parse)

    else:
        print("Error : " + response.text)

    if parse == 'positive':
        happydef = [heart_scenario, takepic1, takepic2]  # songsong , happySong
        # ran = random.randrange(0,4) # 0이상 4미만의 난수
        # print(ran)
        # print(happydef[ran])
        doc_feel.set({'느낌' : 'GOOD', '날짜' : f'{today}'})
        
        ran = random.choice(happydef)
        ran()
    
    elif parse == 'neutral':
        normaldef = [soso_takepic]  # songsong sosoSong,

        doc_feel.set({'느낌' : 'NORMAL', '날짜' : f'{today}'})

        ran = random.choice(normaldef)
        ran()

    else:
        saddef = [touch_scenario1, touch_scenario2]  # songsong , sadSong

        doc_feel.set({'느낌' : 'BAD', '날짜' : f'{today}'})

        ran = random.choice(saddef)
        ran()
    


Start()

#takepic1()
# Cam()

# soso_takepic()

#text_to_speech("안녕, 나는 은쪽이라고해! 너는 이름이 뭐야?")

# getName()
