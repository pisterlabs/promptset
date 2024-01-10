from openai import OpenAI


def get_gpt_result(type, content):
    print("start")

    # open ai private key
    client = OpenAI(
        api_key=
    )

    question = ("너는 지금 computer science 분야에 권위있는 교수야 컴퓨터공학과 학생들이 computer science에 관한 질문이 생겨서 질문하려고 해" +
                "질문 주제는" + type + "이고," +
                "학생들의 질문은" + content + "야" +
                "해당 질문에 대해 대학 학부생들에게 이해하기 쉽게 간단 명료히 설명해 줘")

    print("model에 전달 함수 실행")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        max_tokens=1024,
        model="gpt-3.5-turbo",
    )

    print("end")
    return chat_completion.choices[0].message
