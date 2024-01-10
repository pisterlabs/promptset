import openai

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

# OpenAI 챗봇 모델에 메시지를 보내고 응답을 반환하는 함수
def send_message(message_log):
    # OpenAI의 ChatCompletion API를 사용하여 챗봇의 응답을 얻습니다.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        messages=message_log,   # 이전까지의 대화 기록을 딕셔너리 목록으로 제공
        # max_tokens=1200,      # 생성된 응답에서 최대 토큰(단어 또는 서브워드) 수
        # stop=None,            # 생성된 응답에 대한 중지 시퀀스(여기에서는 사용되지 않음)
        temperature=0.5,        # 생성된 응답의 "창의성" (더 높은 온도 = 창의적)
    )

    # 텍스트가 포함된 챗봇의 첫 번째 응답을 찾습니다(일부 응답에는 텍스트가 없을 수 있음).
    for choice in response.choices:
        if "text" in choice:
            return choice.text

    # 텍스트가 포함된 응답이 없는 경우, 첫 번째 응답의 내용(비어 있을 수 있음)을 반환합니다.
    return response.choices[0].message.content

def main():
    # 챗봇에서 받은 메시지로 대화 기록을 초기화합니다.
    message_log = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # "quit"을 입력할 때까지 실행되는 루프를 시작합니다.
    while True:
        # 터미널에서 사용자의 입력을 받습니다. 
        user_input = input("You: ")

        # 사용자가 "quit"을 입력하면 루프를 종료하고 작별 메시지를 출력합니다.
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        # 사용자의 입력을 대화 기록(message_log)에 추가합니다. 
        message_log.append({"role": "user", "content": user_input})
        
        # 대화 기록을 챗봇에게 보내 응답을 받습니다.
        response = send_message(message_log)

        # 챗봇의 응답을 대화 기록에 추가하고 콘솔에 출력합니다.
        message_log.append({"role": "assistant", "content": response})
        print(f"assistant: {response}")

if __name__ == "__main__":
    main()
