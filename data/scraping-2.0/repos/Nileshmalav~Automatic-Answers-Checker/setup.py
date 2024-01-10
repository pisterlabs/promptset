

azure_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
endpoint = "https://nilesh.cognitiveservices.azure.com/"

from openai import OpenAI
OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
client = OpenAI(
              api_key=OPENAI_API_KEY
        )
def my_score(teacher_text,students_texts):
      completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            # {"role": "system", "content": "You are a very strict answer checker, skilled in comparing student answers with teacher answers by matching word by word, compares the number of words to teacher word and give them score in scale of 10, you need to give int score only."},
            {"role": "system", "content": "Use Natural Language Processing (NLP) Models to evaluate student answers with teacher answers and give only score on the scale of 10 giving integer marks and don't provide any feedback"},
              {"role": "user", "content": "teacher answer : "+teacher_text},
            {"role": "user", "content": "student answer : "+students_texts},
          ]
      )
      # print(completion,type(completion))
      return completion.choices[0].message.content