import openai
import time
import os
from dotenv import load_dotenv
import pygame
from pydub import AudioSegment
import urllib.request

load_dotenv()
LANGUAGE = "user_data/language_select.json"

# ChatGPT
# 3.5
openai.api_key = os.getenv("GPT_API")

# NAVER CLOVA
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


class MYOUNGJA():
    import json
    memory_size = 100

    with open('user_data/user_value.json', 'r') as f:
        data = json.load(f)
        nameValue = data["user_name"]
        manWomanValue = data["user_value"]

    def set_memory_size(self, memory_size):
        '''
        클래스 내의 질문 담을 메모리 저장
        :param memory_size: 메모리
        :return:
        '''
        self.memory_size = memory_size

    def gpt_send_anw(self, question: str):
        self.gpt_standard_messages = [{"role": "assistant",

                                       "content": f"You're a assistant robot for senior."
                                                  f"Your name is 명자. Your being purpose is support.  "
                                                  f"Each response from 명자 should end with a word representing (sad), (daily), (moving), (angry)"
                                                  f"answer in less than 50 words."
                                                  f"Your patient's name is {self.nameValue} and {self.manWomanValue} ."},
                                      {"role": "user", "content": question}]

        response = openai.ChatCompletion.create(
            # model="gpt-4",

            model="gpt-3.5-turbo",
            messages=self.gpt_standard_messages,
            temperature=0.8
        )

        answer = response['choices'][0]['message']['content']

        temp = 0
        ans_list = list(answer)

        if len(ans_list) > 4:

            for i in range(0, len(ans_list)):
                if ans_list[i] == '(':
                    temp = i

            ans_emo = []
            ans_re = []

            for k in range(temp + 1, temp + 10):
                if ans_list[k] == ')':
                    break
                ans_emo.append(ans_list[k])
            ans_emotion = "".join(ans_emo)

            for m in range(0, temp):
                ans_re.append(ans_list[m])
            ans_real = "".join(ans_re)
        else:
            ans_emotion = "평범"
            ans_real = ""

        self.gpt_standard_messages.append(
            {"role": "user", "content": question})
        self.gpt_standard_messages.append(
            {"role": "assistant", "content": answer})
        print(answer)
        print(ans_emotion)
        print(ans_real)
        return [ans_emotion, ans_real]

def speak_first_en():
    from time import localtime
    import random

    tm = localtime()

    question_list = ["How are you??",
                     "I'm happy.", "", "", "", "", "", "", "", ""]

    if tm.tm_hour == 7 and tm.tm_min == 00:
        speaking("Good morning")
    elif tm.tm_hour == 22 and tm.tm_min == 00:
        speaking("Good night")
    elif tm.tm_hour == 12 and tm.tm_min == 00:
        speaking("It's time to take medicine.")
    elif 7 < tm.tm_hour < 22 and tm.tm_min == 00:
        random_ans = random.randrange(0, 9)
        if tm.tm_min == 30:
            if random_ans == 1:
                # 말 걸 내용들
                speaking(question_list[0])
            elif random_ans == 2:
                # 말 걸 내용들
                speaking(question_list[1])
    elif tm.tm_hour == 15 and tm.tm_min == 00:
        speaking("Let's go outside")

    time.sleep(3)

def time_alarm():
    from time import localtime
    tm = localtime()
    speaking(f"현재 시각 {tm.tm_hour}시 {tm.tm_min}분 입니다")

def disease_alarm():
    '''
    복약 알림 테스트용
    '''
    import json

    with open('user_data/user_disease.json', 'r') as f:
        data = json.load(f)
        disease_name = data["disease"]
        disease_hour = data["time_hour"]
        disease_min = data["time_min"]
    speaking(f"현재 시각 {disease_hour}시 {disease_min}분, {disease_name}약 드실 시간이예요!")

def speak_first():
    '''
    먼저 말 거는 함수
    :return: X
    '''
    from time import localtime
    import random
    import json

    with open('user_data/user_disease.json', 'r') as f:
        data = json.load(f)
        disease_name = data["disease"]
        disease_hour = data["time_hour"]
        disease_min = data["time_min"]
    tm = localtime()

    question_list = ["오늘 몸 상태는 어때요?",
                     "저는 오늘도 행복해요. 오늘 어떠세요??", "", "", "", "", "", "", "", ""]

    if tm.tm_hour == 7 and tm.tm_min == 00:
        speaking("좋은 아침이에요!! 오늘도 좋은 하루 되세요!!")
    elif tm.tm_hour == 22 and tm.tm_min == 00:
        speaking("이제 잘 시간이에요!! 편안한 밤 되세요!!")
    elif tm.tm_hour == disease_hour and tm.tm_min == disease_min:
        speaking(f"{disease_name}약 드실 시간이에요!")
    elif 7 < tm.tm_hour < 22 and tm.tm_min == 00:
        random_ans = random.randrange(0, 9)
        if tm.tm_min == 30:
            if random_ans == 1:
                # 말 걸 내용들
                speaking(question_list[0])
            elif random_ans == 2:
                # 말 걸 내용들
                speaking(question_list[1])
    elif tm.tm_hour == 15 and tm.tm_min == 00:
        speaking("우리 산책 나가요!")

    time.sleep(3)

def speaking(anw_text):
    # NAVER CLOVA
    encText = urllib.parse.quote(anw_text)
    data = f"speaker=ndain&volume=0&speed=0&pitch=0&format=mp3&text=" + encText
    urls = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    requests = urllib.request.Request(urls)
    requests.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    requests.add_header("X-NCP-APIGW-API-KEY", client_secret)
    
    try:
        response = urllib.request.urlopen(requests, data=data.encode('utf-8'))
        rescodes = response.getcode()
        if rescodes == 200:
            print("mp3 저장 완료")
            response_body = response.read()
            with open('./ResultMP3.mp3', 'wb') as f:
                f.write(response_body)

            # Convert MP3 to WAV
            filename = "ResultMP3.mp3"
            dst = "test.wav"
            sound = AudioSegment.from_mp3(filename)
            sound.export(dst, format="wav")

            # Play the WAV file using pygame
            pygame.mixer.init()
            pygame.mixer.music.load(dst)
            pygame.mixer.music.play()

            # Wait for the music to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            print("Error in audio retrieval: HTTP Response Code", rescodes)

    except Exception as e:
        print("Error occurred while processing audio: ", e)

    finally:
        # Clean up: remove created audio files
        if os.path.exists("ResultMP3.mp3"):
            os.remove("ResultMP3.mp3")
        if os.path.exists("test.wav"):
            os.remove("test.wav")

def speaking_en(anw_text):
    # NAVER CLOVA
    encText = urllib.parse.quote(anw_text)
    data = f"speaker=djoey&volume=0&speed=0&pitch=0&format=mp3&text=" + encText
    urls = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
    requests = urllib.request.Request(urls)
    requests.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    requests.add_header("X-NCP-APIGW-API-KEY", client_secret)
    
    try:
        response = urllib.request.urlopen(requests, data=data.encode('utf-8'))
        rescodes = response.getcode()
        if rescodes == 200:
            print("mp3 저장 완료")
            response_body = response.read()
            with open('./ResultMP3.mp3', 'wb') as f:
                f.write(response_body)

            # Convert MP3 to WAV
            filename = "ResultMP3.mp3"
            dst = "test.wav"
            sound = AudioSegment.from_mp3(filename)
            sound.export(dst, format="wav")

            # Play the WAV file using pygame
            pygame.mixer.init()
            pygame.mixer.music.load(dst)
            pygame.mixer.music.play()

            # Wait for the music to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        else:
            print("Error in audio retrieval: HTTP Response Code", rescodes)

    except Exception as e:
        print("Error occurred while processing audio: ", e)

    finally:
        # Clean up: remove created audio files
        if os.path.exists("ResultMP3.mp3"):
            os.remove("ResultMP3.mp3")
        if os.path.exists("test.wav"):
            os.remove("test.wav")



def mic_en(time):
    import requests
    import sounddevice as sd
    from scipy.io.wavfile import write

    # 음성 녹음
    fs = 44100
    seconds = time

    myRecording = sd.rec(int(seconds * fs), samplerate=fs,
                         channels=4)  # channels는 마이크 장치 번호
    print("녹음 시작")
    # 마이크 장치 번호 찾기 => python -m sounddevice
    sd.wait()
    write('sampleWav.wav', fs, myRecording)

    # Voice To Text => 목소리를 텍스트로 변환
    # 기본 설정
    lang = "Eng"  # 언어 코드 ( Kor, Jpn, Eng, Chn )
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang

    # 녹음된 Voice 파일
    data_voice = open('sampleWav.wav', 'rb')

    # 사용할 header
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/octet-stream"
    }

    # VTT 출력
    response = requests.post(url, data=data_voice, headers=headers)

    result_man = str(response.text)
    result = list(result_man)
    count_down = 0
    say_str = []

    for i in range(0, len(result) - 2):
        if count_down == 3:
            say_str.append(result[i])

        if response.text[i] == "\"":
            if count_down == 3:
                break
            else:
                count_down += 1

    anw_str = ''.join(map(str, say_str))

    print(anw_str)

    return anw_str


def mic(time):
    '''
    주어진 음성을 마이크로 녹음 후 문장으로 변환
    :return: 음성 파일 문장으로 변환시켜 넘김
    '''
    import requests
    import sounddevice as sd
    from scipy.io.wavfile import write

    # 음성 녹음
    fs = 44100
    seconds = time

    myRecording = sd.rec(int(seconds * fs), samplerate=fs,
                         channels=4)  # channels는 마이크 장치 번호
    print("녹음 시작")
    # 마이크 장치 번호 찾기 => python -m sounddevice
    sd.wait()
    write('sampleWav.wav', fs, myRecording)

    # Voice To Text => 목소리를 텍스트로 변환
    # 기본 설정
    lang = "Kor"  # 언어 코드 ( Kor, Jpn, Eng, Chn )
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang

    # 녹음된 Voice 파일
    data_voice = open('sampleWav.wav', 'rb')

    # 사용할 header
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/octet-stream"
    }

    # VTT 출력
    response = requests.post(url, data=data_voice, headers=headers)

    result_man = str(response.text)
    result = list(result_man)
    count_down = 0
    say_str = []

    for i in range(0, len(result) - 2):
        if count_down == 3:
            say_str.append(result[i])

        if response.text[i] == "\"":
            if count_down == 3:
                break
            else:
                count_down += 1

    anw_str = ''.join(map(str, say_str))

    print(anw_str)

    return anw_str

def name_check_en():
    import json
    global common
    common = 0
    with open('user_data/user_value.json', 'r') as f:
        data = json.load(f)
        if data["user_name"] == "":
            speaking_en("Hello sir. I have no data.so I ask you something.")
            speaking_en("What's your name?")
            name_ = mic(2)
            speaking_en(f"Hi! {name_}")
            speaking_en("What's your gender. please speak he or she.")
            manWoman = mic(2)
            if manWoman == "he":
                manWoman_ = "he"
            elif manWoman == "she":
                manWoman_ = "she"
            else:
                while manWoman != "he" or "she":
                    speaking_en("I'm sorry. could you repeat please?")
                    manWoman = mic(2)
                    if manWoman == "he":
                        manWoman_ = "he"
                        break
                    elif manWoman == "she":
                        manWoman_ = "she"
                        break
            common = 1
            speaking_en("Thank you. setting is over")
        else:
            name_ = data["user_name"]
            manWoman_ = data["user_value"]

        write_data = {
            "user_name": f"{name_}",
            "user_value": f"{manWoman_}"
        }

        if common == 1:
            with open('./user_value.json', 'w') as d:
                json.dump(write_data, d)

    return [name_, manWoman_]

def name_check():
    import json
    global common
    common = 0
    with open('user_data/user_value.json', 'r') as f:
        data = json.load(f)
        if data["user_name"] == "":
            use_sound("./mp3/first_0.wav")
            use_sound("./mp3/first_set_2.wav")
            name_ = mic(2)
            speaking(f"안녕하세요! {name_}님")
            use_sound("./mp3/first_set_3.wav")
            manWoman = mic(2)
            if manWoman == "남자":
                manWoman_ = "he"
            elif manWoman == "여자":
                manWoman_ = "she"
            else:
                while manWoman != "남자" or "여자":
                    use_sound("./mp3/first_set_4.wav")
                    manWoman = mic(2)
                    if manWoman == "남자":
                        manWoman_ = "he"
                        break
                    elif manWoman == "여자":
                        manWoman_ = "she"
                        break
            common = 1
            use_sound("./mp3/first_set_done.wav")
        else:
            name_ = data["user_name"]
            manWoman_ = data["user_value"]

        write_data = {
            "user_name": f"{name_}",
            "user_value": f"{manWoman_}"
        }

        if common == 1:
            with open('user_data/user_value.json', 'w') as d:
                json.dump(write_data, d)

    return [name_, manWoman_]


def name_ini():
    import json
    write_data = {
        "user_name": "",
        "user_value": ""
    }
    with open('user_data/user_value.json', 'w') as d:
        json.dump(write_data, d)


def use_sound(loc):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(loc)
        pygame.mixer.music.play()

        # Wait for the music to play before exiting
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error occurred while playing sound: {e}")


def langugage_change(kr):
    import json
    if kr:
        # 한국어
        write_data = {
            "language": 1
        }
        with open(LANGUAGE, 'w') as lang:
            json.dump(write_data, lang)
    elif kr == False:
        # 영어
        write_data = {
            "language": 0
        }
        with open(LANGUAGE, 'w') as lang:
            json.dump(write_data, lang)

def language_check():
    import json
    with open(LANGUAGE, 'r') as lang:
        data = json.load(lang)
        if data["language"] == 1:
            return 1
        else:
            return 0
        
