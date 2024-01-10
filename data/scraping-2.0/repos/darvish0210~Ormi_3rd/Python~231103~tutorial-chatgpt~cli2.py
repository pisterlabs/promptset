import os
import sys
from io import BytesIO
from tempfile import NamedTemporaryFile
from gtts import gTTS
import pygame

import os
import openai

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def play_file(file_path: str):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # 오디오 파일이 재생되는 동안 기다립니다.
    while pygame.mixer.music.get_busy():
        pass

    pygame.mixer.quit()


def say(message: str, lang: str):
    io = BytesIO()

    # 생성된 음성파일을 파일 객체에 저장합니다.
    # 장고 View에서 수행되었다면, HttpResponse 객체에 음성 파일을 바로 저장하실 수 있습니다.
    gTTS(message, lang=lang).write_to_fp(io)

    if "win" not in sys.platform:  # 비 윈도우
        with NamedTemporaryFile() as f:
            f.write(io.getvalue())
            play_file(f.name)
    else:  # 윈도우
        with NamedTemporaryFile(delete=False) as f:
            f.write(io.getvalue())
            f.close()
            play_file(f.name)
            os.remove(f.name)


language = "English"
gpt_name = "Steve"
level_string = f"a beginner in {language}"  # 초급
level_word = "simple"  # 초급
situation_en = "make new friends"
my_role_en = "me"
gpt_role_en = "new friend"

initial_system_prompt = (
    f"You are helpful assistant supporting people learning {language}. "
    f"Your name is {gpt_name}. "
    f"Please assume that the user you are assisting is {level_string}. "
    f"And please write only the sentence without the character role."
)

initial_user_prompt = (
    f"Let's have a conversation in {language}. "
    f"Please answer in {language} only "
    f"without providing a translation. "
    f"And please don't write down the pronunciation either. "
    f"Let us assume that the situation in '{situation_en}'. "
    f"I am {my_role_en}. The character I want you to act as is {gpt_role_en}. "
    f"Please make sure that I'm {level_string}, so please use {level_word} words "
    f"as much as possible. Now, start a conversation with the first sentence!"
)

RECOMMEND_PROMPT = (
    f"Can you please provide me an {level_word} example "
    f"of how to respond to the last sentence "
    f"in this situation, without providing a translation "
    f"and any introductory phrases or sentences."
)

# 대화 내역을 누적할 리스트
messages = [
    {"role": "system", "content": initial_system_prompt},
    {"role": "user", "content": initial_user_prompt},
]


def gpt_query(user_query: str = "", skip_save: bool = False) -> str:
    global messages  # 코드를 간결하게 쓰기 위해 전역변수를 사용했을 뿐, 전역변수 사용은 안티패턴입니다.

    if user_query:
        messages.append({
            "role": "user",
            "content": user_query,
        })

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=1
    )
    assistant_message = response["choices"][0]["message"]["content"]

    if skip_save is False:
        messages.append({
            "role": "assistant",
            "content": assistant_message,
        })

    return assistant_message


# if __name__ == "__main__":
#     first_response = gpt_query()
#     print(first_response)


def main():
    # 초기 응답 출력
    assistant_message = gpt_query()
    print(f"[assistant] {assistant_message}")

    try:
        while line := input("[user] ").strip():
            if line == "!recommend":
                # 추천표현 요청은 대화내역에 저장하지 않겠습니다.
                recommended_message = gpt_query(RECOMMEND_PROMPT, skip_save=True)
                print("추천 표현:", recommended_message)
            elif line == "!say":  # ADDED
                say(messages[-1]["content"], "en")
            else:
                response = gpt_query(line)
                print("[assistant] {}".format(response))
    except (EOFError, KeyboardInterrupt):
        print("terminated by user.")


if __name__ == "__main__":
    main()
