import openai
import re
import config

# 발급받은 API 키 설정
OPENAI_API_KEY = config.OPEN_AI_API_KEY

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

# 모델 - GPT 3.5 Turbo 선택 / 고정 질문
model = "gpt-3.5-turbo"
query = "아래 내용을 한글로 요약해줘 \n"

# 퀴즈 질문
STATIC_QUESTION = "아래는 Chat GPT를 통해 뉴스 기사를 요약한 내용입니다. 해당 __에 들어갈 말은 무엇일까요? \n\n"

# 퀴즈를 위한 자음 리스트
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 퀴즈 만들기
def make_quiz_and_answer(origin):
    answer = origin[1]
    temp = "_" * len(answer)
    quiz = origin[0].replace(answer, temp)

    hint_list = list()
    for a in answer:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', a) is not None:
                char_code = ord(a) - BASE_CODE
                char1 = int(char_code / CHOSUNG)
                hint_list.append(CHOSUNG_LIST[char1])

    hint = ""
    for h in hint_list:
        hint = hint+str(h)
        
    print(hint)
    
    return {"quiz" : STATIC_QUESTION + quiz, "answer" : answer, "hint" : hint}

# ChatGPT API
def chatgpt_quiz(text):

    # 1. ChatGPT에게 기사 내용 요약 요청
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query+text}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    summary = response['choices'][0]['message']['content']

    # 2. ChatGPT에게 요약 문장에서 중요한 단어를 출력
    messages.append(
        {"role": "assistant", "content": summary}
    )
    messages.append(
        {"role": "user", "content": "위 내용에서 가장 중요한 단어를 1개 찾아줘"}
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    word = response['choices'][0]['message']['content'].replace(" ", "")

    origin = [summary, word]
    return make_quiz_and_answer(origin)
