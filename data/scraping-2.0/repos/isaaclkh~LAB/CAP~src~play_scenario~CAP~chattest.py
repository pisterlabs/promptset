import openai
import os

# Load OpenAI API key from environment variable
openai.api_key = "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"

pre = "나에게 성경구절 하나를 추천해 줘. 너는 대답을 이런 형식으로 해야해."
rule1 = "1) 반말(~야)을 사용해줘. 친구에게 말을 하듯이 대답해줘."
rule2 = "2) 대답을 시작할 때, 말씀의 제목을 먼저 말해줘. 예를 들어 '이사야 5장 5절이야'와 같은 형태로 대답을 시작해줘."
rule3 = "3) 성경 구절은 따로 문단을 나눠 대답을 해줘, 마지막에는 한 문장으로 위로를 해줘"
rule4 = "4) 나를 부를 때는 '너' 라고 불러줘."
rule5 = "5) 질문은 하지 마."
rule6 = "6) 시편 같은 경우에는 장이 아니라 편이야 위에와 같은 형식으로 꼭 대답을 해."
ex = "아래는 예시를 보여줄게. 같은 형식으로 대답을 해야해. [(창세기 28장 15절)이야. \"보라, 나는 너와 함께 있어서 네가 가는 모든 길에서 너를 지키리니 이르기를 내가 너를 보내지 아니하고 네게 허락한 땅으로 돌아가게 하리라 할 때까지\" 너가 잃어버린 것이 얼마나 아까워서 불안하고 슬프겠지만, 하나님은 네가 가는 길에서 너를 지키시며, 네가 돌아가는 땅까지 너를 인도해주시리라 믿어봐.] 위의 예시처럼 대답을 해줘."
rule7 = "7) 대답을 할 때 존댓말을 절대 사용하지마."
rule8 = "8) 말씀 주소와 말씀을 먼저 말해줘."

# Generate text using the GPT model
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    n=1,
    messages=[
        {"role": "system", "content": "너는 정말 좋은 친구야."},
        {"role": "user", "content": "오늘 배가 너무 고팠어" + pre + rule1 + rule2 + rule3 + rule4 + rule5 + rule6 + ex + rule7 + rule8},
    ])

message = response.choices[0]['message']

print("{}: {}".format(message['role'], message['content']))