import openai
import nltk
from nltk import word_tokenize
from autocorrect import Speller
from nltk.stem.wordnet import WordNetLemmatizer

#사용시 키 입력
openai.api_key = ''

#사용자의 prompt를 토대로 category 및 detail을 추천해주는 함수
def request(text) :
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": text},
      ]
  )
  return response.choices[0].message.content

#object 종류가 늘어남에 따라 변경해야 함
'''target = 'chair'

# object가 포함되어 있으면 object를 반환, 없으면 -1을 반환
def checkObject(prompt) :
  spell = Speller(lang = 'en')
  lemmatizer = WordNetLemmatizer()
  words = word_tokenize(prompt)
  for word in words :
    corrected_word = lemmatizer.lemmatize(spell(word), 'v')
    if target == corrected_word :
      return corrected_word
  return '-1'
'''

import json

#사용해야 하는 것 / input : 사용자 input, output : json
def getAnswer(prompt) :
  obj = checkObject(prompt)
  if obj == '-1' :
    return -1
  recommend = request("I want to express " + obj + ". Is there anything I can describe in more detail? Please answer in the format below\n1. Category \n2. Category \n3. Category \n4. Category \n5. Category")
  detail = request("I want to express " + prompt + ". Recommend a some example with detailed description? Please answer in the format below\n1.xxx : detail\n2.xxx : detail\n3.xxx : detail\n4.xxx : detail\n5.xxx : detail\n")
  detail = detail.replace('\n\n','\n')
  print(detail)
  details = detail.split('\n')

  json_object = {
      "recommend" :recommend
  }

  index = 0
  detail_json = {}

  for d in details :
    detail_json['detail' + str(index)] = {"prompt" : d.split(':')[0], "detail" : d.split(':')[1]}
    index += 1
  
  json_object['detail'] = detail_json
  json_string = json.dumps(json_object)
  return json_string