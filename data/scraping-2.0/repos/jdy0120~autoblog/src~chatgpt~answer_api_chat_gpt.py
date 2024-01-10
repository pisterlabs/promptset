import os
import openai
from dotenv import load_dotenv
from src.chatgpt.slice_token import truncate_text

load_dotenv(verbose=True)

# openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("CHAT_GPT_API")

def answer_api_chat_gpt(shopinfo,text,length=0,model='gpt-3.5-turbo',type='normal'):
  
  question = truncate_text(text)
  
  if (type == 'normal'):
    try:
      response = openai.ChatCompletion.create(
        model=model, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages=[{"role": "user", "content": f"다음은 {shopinfo}의 리뷰를 모은 글이야 {str(length) + '토큰 이하로' if length != 0 else ''} {shopinfo} 업체의 정보에 대한 장단점, 음식에 대한 리뷰 등을 지인이 {shopinfo}를 소개해주는 방식으로 최대한 핵심만 자세하게 설명해줘 \n {question}"}]
      )
    except Exception as e:
      print('gpt error',e)
      return ''
  
  elif(type == 'information'):
    try:
      response = openai.ChatCompletion.create(
        model=model, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages=[{"role": "user", "content": f"다음은 {shopinfo}의 정보야 읽기쉽게 정리해서 출력해줘 \n {question}"}]
      )
    except Exception as e:
      print('gpt error',e)
      return ''
    
  else:
    try:
      response = openai.ChatCompletion.create(
        model=model, # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        messages=[{"role": "user", "content": f"다음은 {shopinfo}의 리뷰를 모은 글이야 {shopinfo} 업체의 정보에 대한 장단점, 음식에 대한 리뷰 등을 지인이 {shopinfo}를 소개해주는 방식으로 5000자에 맞춰 최대한 자세하게 설명해주고 제일 앞에 평점도 별을 이용해 추가시켜줘 \n {question}"}]
      )
    except Exception as e:
      print('gpt error',e)
      return ''
  
  answer = response.choices[0].message.content
  file_path = './shopdata/{0}.txt'.format(shopinfo)

  with open(file_path, 'a', encoding='utf-8') as f:
    f.write('\n' + answer)
    f.close()
  
  return answer