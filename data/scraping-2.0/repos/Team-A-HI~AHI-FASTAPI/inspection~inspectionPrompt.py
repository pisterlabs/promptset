# from openai import OpenAI
# from configset.config import getAPIkey ,getModel
# import openai
# import re

# client = OpenAI()
# OPENAI_API_KEY = getAPIkey()
# MODEL = getModel()


# def post_gap(system_content, user_content):
#     try:
#         openai.api_key = OPENAI_API_KEY
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages= [
#                 {"role" : "system","content" : system_content},
#                 {"role" : "user", "content" : user_content} 
#             ],
#             stop=None,
#             temperature=0.5
#         )
#         answer = response.choices[0].message.content
#         print("gpt 답변 : " + answer)
#         return answer
#     except Exception as e:
#         resp ={
#             "status" : e,
#             "data" : "그냥 오류요 뭐요 다시 시도해보든가"
#         }
#         return {"resp" : resp}


# def create_prediction_prompt(ask):
#     title = ask.title
#     if ask.title == '검수':
#         system_content = "You're the best inspector."
#     else :
#         system_content = "You're the best resume editor"
#     ask = f"자기소개서 제목은{ask.introductionTitle}이고; 내용은{ask.content}이고;{title}방향은 {ask.keyword}으로 진행해줘"
#     pre_prompt = f"한국어로 답변해줘; 자기소개서를 {title}해줘; 원본과 {title}내용을 분리해서 보여줘; \n\n"
#     print(f"system_content : {system_content}")
#     print(f"ask : {ask}")
#     print(f"pre_prompt : {pre_prompt}")
#     answer = post_gap(system_content , pre_prompt + ask)
#     sentences = re.split(r'\d+\.',answer)
#     sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
#     print(f'sentences : {sentences}')

#     return sentences

