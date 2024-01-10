from django.http import JsonResponse
from mv_backend.lib.database import Database
from mv_backend.settings import OPENAI_API_KEY
from mv_backend.lib.common import CommonChatOpenAI
from mv_backend.api.function.retrieve import *
from mv_backend.api.function.reflect import *
from mv_backend.api.function.cefr import *
import json, openai

db = Database()

openai.api_key = OPENAI_API_KEY

chat = CommonChatOpenAI()

#프론트에게 유저 프로필을 전달하기 위한 함수
#프론트에게 전달받은 유저 이름을 기반으로, DB에서 누적된 정보(관심사, 대화 성향, 미진 사항)를 가져온다.
#관심사, 대화 성향, 미진 사항은 각각 개수에 따라 내림차순으로 정렬된다(중복된 값을 앞으로).
#프론트와 통신하는 call 함수
def call(request):
    #decoding
    body_unicode = request.body.decode('utf-8')
    
    #json -> dict
    body = json.loads(body_unicode)

    #user_name 값 가져오기
    user_name = ""
    for chat_data in body["messages"]:
        if chat_data["role"] == "user_name":
            user_name = chat_data["content"]
            break
    
    #reflect, retrieve, cefr 정보 DB에서 가져오기
    reflect = Database.get_recent_documents(db, user_name, "Reflects_Kor", 3)
    retrieve = Database.get_recent_documents(db, user_name, "Retrieves_Kor", 3)
    cefr = Database.get_recent_documents(db, user_name, "CEFR_GPT", 1)
    
    interest_dict = dict()
    coversationStyle_dict = dict()
    cefr_string = ""
    
    data_num = 0
    #reflect 정보에서 관심사와 대화 성향을 추출한다(출력 format를 split하여 활용).
    #dict를 활용하여, key는 관심사 및 대화성향 | value는 개수를 카운트한다.
    for i in reflect:
        result = i["reflect"].split(":")
        interests = result[1].split('\n')[0].split(",")
        for interest in interests:
            if interest not in interest_dict.keys():
                interest_dict[interest] = 1
            else:
                interest_dict[interest] += 1
        conversationStyles = result[2].split('\n')[0].split(",")
        if (data_num == 0):
            for conversationStyle in conversationStyles:
                if conversationStyle not in coversationStyle_dict.keys():
                    coversationStyle_dict[conversationStyle] = 1
                else:
                    coversationStyle_dict[conversationStyle] += 1
        data_num += 1
    
    data_num = 0
    
    #retrieve 정보에서 미진사항을 추출한다(출력 format를 split하여 활용).
    #dict를 활용하여, key는 미진 사항 | value는 개수를 카운트한다.
    retrieve_dict = dict()
    for i in retrieve:
      result = i["retrieve"].split("이유:")
      print(result)
      length = len(result)
      for j in range(length-1):
        retrieve_result = result[j+1].split('\n')[0]
        if retrieve_result not in retrieve_dict.keys():
            retrieve_dict[retrieve_result] = 1
        else:
            retrieve_dict[retrieve_result] += 1
    for i in cefr:
      cefr_string += i["cefr"]
    
    #개수(vaule)에 따라, 내림차순으로 각각 정렬
    sorted_interest = list(dict(sorted(interest_dict.items(), key = lambda item: item[1], reverse = True)).keys())
    sorted_conversationStyle = list(dict(sorted(coversationStyle_dict.items(), key = lambda item: item[1], reverse = True)).keys())
    sorted_retrieve = list(dict(sorted(retrieve_dict.items(), key = lambda item: item[1], reverse = True)).keys())
    
    #프론트에 보내줄 message list에 각각 append한다.
    messages_response = [
        {
            "role": "interest",
            "content": str(sorted_interest).removeprefix('[').removesuffix(']').replace("'", "")
        }
    ]
    
    messages_response += [
        {
            "role": "conversationStyle",
            "content": str(sorted_conversationStyle).removeprefix('[').removesuffix(']').replace("'", "")
        }
    ]

    messages_response += [
        {
            "role": "retrieve",
            "content": str(sorted_retrieve).removeprefix('[').removesuffix(']').replace("'", "")
        }
    ]

    messages_response += [
        {
            "role": "cefr",
            "content": cefr_string
        }
    ]

    print(messages_response)
    
    return JsonResponse({
        "messages": messages_response
    })