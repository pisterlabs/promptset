import openai
import sys

openai.api_key = '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'

input = sys.argv[1]

completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role":"system", "content":"두가지의 이유와 결론을 통해 보이스피싱인지 아닌지 의심됨/의심되지 않음 으로 답변해줘."},
        {"role":"user","content": "엄마 나 다쳤어. 10만원만 보내줘. 왜? 괜찮아 아들? 응 괜찮아 농협은행 010238493298로 보내주면 돼. 알았어 아들"},
        {"role":"assistant","content":" 이유 : 갑작스러운 사건, 계좌번호 제공, 결론 : 의심됨."},
        {"role":"user", "content":input}
      ]
)

chat_response = completion.choices[0].message.content
print(chat_response)