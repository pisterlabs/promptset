from flask import Flask, request, jsonify
from openai import OpenAI
import openai
import os

app = Flask(__name__)

# OpenAI API 키를 환경변수에서 가져옵니다.
# openai.api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()
# Assistant를 생성합니다.
assistant = client.beta.assistants.create(
    name = "Math Tutor",
    model="gpt-3.5-turbo",  # 사용할 모델을 지정합니다.
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],  # 필요한 도구를 지정합니다.
)

@app.route('/ask_math', methods=['POST'])
def ask_math():
    data = request.json
    user_message = data['message']

    # 새로운 대화를 위한 Thread를 생성합니다.
    thread = client.beta.threads.create()

    # 사용자의 메시지를 Thread에 추가합니다.
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    try:
        # Assistant를 실행합니다.
        run = client.beta.threads.runs.create(
            assistant_id=assistant.id,
            thread_id=thread.id,
        )

        # Assistant의 응답을 받아옵니다.
        while True:
            if run.status == "completed":
                break
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        # Assistant의 응답을 확인합니다.
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
      
        # 사용자에게 보낼 메시지를 결정합니다.
        if messages.data:
            # value 값을 저장할 변수
            response_content = ""

            # messages.data의 각 ThreadMessage 객체에 대해 반복
            for message in messages.data:
                # 각 메시지의 content 속성에 대해 반복
                for content in message.content:
                    if content.type == 'text':
                        # 'text' 유형의 content에서 'value' 값을 추출하여 저장
                        response_content = content.text.value
                        break  # 첫 번째 'text' 유형의 content를 찾으면 반복 중단
                if response_content:
                    break  # 'value' 값을 찾으면 바깥쪽 반복도 중단
        
            return jsonify({'response': response_content}), 200

    except Exception as e:
        # 로그에 에러를 출력합니다.
        app.logger.error(f'An error occurred: {str(e)}')
        return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/')
def index():
    return app.send_static_file('chat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
