import openai
from config import OPENAI_KEY
from y_data import y_data

openai.api_key = OPENAI_KEY




def check_rate(search_ward):
    y = y_data(search_ward)
    data_list = y.get_video_data()
    title_list = []
    for data in data_list:
        title_list.append(data['YouTubeBigBoxs'][0]['title'])
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You can't print anything other than numbers and commas."},
        {"role": "system", "content": "You must start with zero when you index"},
        {"role": "system", "content": "You should indicate the index of the first list you received when sorting."},
        {"role": "user", "content": f"{title_list}의 제목들이 있는 데 이걸 기준으로한 인덱스 숫자를 사용하여 {search_ward}와 관련도가 높은 순서로 정렬한 결과를 인덱스 숫자로 알려줘"},
    ],
    temperature = 0.1,
    frequency_penalty = 0
    )
    re = response['choices'][0]['message']['content']
    idx_list = re.split(',')
    sorted_list = []
    for idx in idx_list:
        sorted_list.append(int(idx))
    result = []
    try:
        for i in sorted_list:
            result.append(data_list[i])
        result = result[0:5]
        return result
    except:
        return {"messeage" : "gpt-3.5-turbo error please register your payment."}

if __name__ == '__main__':
    r = check_rate('뉴진스')
    print(r)



