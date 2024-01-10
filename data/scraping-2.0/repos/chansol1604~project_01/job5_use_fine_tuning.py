from openai import OpenAI
client = OpenAI(
    api_key="")

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:personal::8YQtjK9S",
  messages=[
    # {"role": "system", "content": "You are a helpful assistant."},
    # {"role": "system", "content": "당신은 반말을 사용하는 인공지능 챗봇입니다. 입력된 일기를 읽고, 친구에게 써주듯이 코멘트를 작성해주세요."},
    # {"role": "system", "content": "당신은 친절한 인공지능 챗봇입니다. 입력된 일기를 읽고,상담사가 상담자에게 상담해주듯이 코멘트를 친절하게 작성해주세요."},
    # {"role": "system", "content": "당신은 sns어체를 사용하는 인공지능 챗봇입니다. 사람들이 sns에서 쓰는 어체로 글에 대해 코멘트 해주세요."},
    {"role": "system", "content": "당신은 sns어체를 사용하는 인공지능 챗봇입니다."},
    #{"role": "user", "content": "제목:강아지 오늘 공원을 산책하는 도중에 귀여운 강아지를 보았다. 그 강아지는 지금까지 본 강아지 중 가장 귀여운 강아지였다. 그 강아지를 몇 번 쓰다듬었고, 강아지의 초롱초롱한 눈빛이 아주 인상깊었다. 강아지 덕분에 스트레스도 풀고 힐링할 수 있었다. 다음에 기회가 된다면 꼭 강아지를 키우고 싶다."}
    # {"role": "user", "content": "제목:시험   오늘 중간고사를 보고 왔다. 나는 시험 한 달 전부터 아주 열심히 했는데 결과가 생각보다 좋게 나오지 않았다. 수학은 90점을 맞았지만, 영어는 70점 밖에 맞지 못했다. 또한, 다른 과목은 평균이 60점 밖에 되지 않았다. 너무 슬프다. 앞으로는 1달 반 전부터 시험을 준비해서 더 성적을 높여야 겠다. 그래서 꿈을 꼭 이뤄야 겠다."}
    {"role": "user", "content": "오늘 일본 여행을 갔는데 너무 좋았어"}
  ]
)

print(completion.choices[0].message)
