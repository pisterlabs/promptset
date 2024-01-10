import openai

openai.api_key = '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'

#사용자의 prompt를 토대로 category 및 detail을 추천해주는 함수
def recommeded(discription) :
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "I want to express " + discription + ". Is there anything I can describe in more detail?"},
      ]
  )
  return response.choices[0].message.content
