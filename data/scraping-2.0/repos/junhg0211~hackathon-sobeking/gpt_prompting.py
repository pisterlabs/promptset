import openai
import json

from util import get_const


def database_stringify(datetime_set):
    with open ("date_data.json", "r", encoding='utf-8') as f:
        data = json.load(f)

    output = str()

    if datetime_set == 1:
        for i in range(len(data) + 1):
            if i == 0:
                output += "소비날짜,소비금액,소비물품\n"
            else:
                output += f"{data[i-1]['consume_at']},{data[i-1]['amount']},{data[i-1]['content']}\n"

    elif datetime_set == 2:
        for i in range(8):
            if i == 0 :
                output += "소비날짜,소비금액,소비물품\n"
            else :
                output += f"{data[i-1]['consume_at']},{data[i-1]['amount']},{data[i-1]['content']}\n"

    return output


def init(datetime_set) :
    openai.api_key = get_const('openai_key')
    month_or_week = {1: "한 달", 2: "일주일"}

    messages = [
        {
            "role": "system",
            "content": f"당신은 소비 패턴 분석가의 역할을 수행해주시면 됩니다. {month_or_week[datetime_set]}의 소비 내역이 입력되면, 소비 내역 중 잦은 소비를 갖는 소비 패턴(카드 제외)과 50000원 이상의 모든 특이 소비를 분별해주십시오. 그리고, 그 주에 대한 소비에 대한 피드백을 남겨주셨으면합니다. 마지막으로, 소비에 있어서 칭찬할 점을 말해주십시오."      
        },
        {
            "role": "system",
            "content": f"부가적인 설명은 배제하고, '1.일정한 소비 패턴, 2.50000원 이상의 특이 소비,3.소비에 대한 피드백, 4.소비에 있어서 칭찬할 점'만을 출력해주십시오."      
        }
    ]
    return messages


async def ask(datetime_set) :
    peak_threshold = 50000
    # datetime_set = int(input())  # 1 = week, 2 = day
    messages = init(datetime_set)
    userMessage = database_stringify(datetime_set) #소비에 대한 쿼리문 읽어오기
    messages.append(
        {
            "role": "user",
            "content": userMessage
        }
    )

    response = await openai.ChatCompletion.acreate(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    botMessage = response.choices[0].message.content

    return botMessage


if __name__ == '__main__':
    ask(1)
