import openai
import os

# API 키 직접 설정
openai.api_key = "your_api_key"


def is_hotel_related(question):
    """
    사용자의 질문이 호텔 추천과 관련되었는지 GPT API를 사용하여 판단합니다.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question},
            {"role": "system", "content": "이 질문의 의도가 호텔과 연관이 있으면 1을, 아니면 0으로 답변해주세요."},
        ],
    )

    # API 응답에서 '1' 또는 '0'만 추출하여 반환
    return int(response.choices[0].message["content"].strip())


# main 함수
if __name__ == "__main__":
    question = "호텔 추천해줘"
    result = is_hotel_related(question)
    print(f"호텔 관련 질문: {result}")
