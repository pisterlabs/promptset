import openai

import openai
import os
import pandas as pd
import cx_Oracle

openai.api_key = "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"

def chatbot(content,model="gpt-3.5-turbo",messages=[],temperature=0):
    try:
        messages.append({'role': 'system', 'content': 'You are an expert on nutritional supplements and health functional foods and I am a cubot to help your health if you do not ask questions about health functional foods. Just ask me questions about health functional foods. You have to answer in Korean. If you ask questions about nutritional supplements and health functional foods, find them in the information in foodtable_data. If the user wants to recommend nutritional supplements or health functional foods, health concerns (e.g., fatigue, brain activity, eye health, body fat, blood vessels and blood circulation, liver health, intestinal health, stress and sleep, immune function, blood cholesterol, bone health, aging and stomach food, thyroid health, female health, male, anemia, blood sugar, blood, blood, blood and blood and maltreature, 5 healthy skin, healthy, healthy, healthy, gum and cartilage, gum, and respiration. But do not tell me the nutrients. And the number of letters is up to 500 characters.'})
        messages.append({'role':'user','content':content})
        response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature
        )
        answer = response.choices[0].message.content
        messages.append({'role':'assistant','content':answer})
        findKeyword(answer)
        return {'status':'SUCCESS','messages':messages}
    except openai.error.APIError as e:
         print(f'OpenAI API returned an API Error: {e}')
         return {'status':'FAIL','messages':e}
    except openai.error.APIConnectionError as e:
        print(f'Failed to connect to OpenAI API: {e}')
        return {'status': 'FAIL', 'messages': e}
    except openai.error.InvalidRequestError as e:
        print(f'Invalid Request to OpenAI API: {e}')
        return {'status': 'FAIL', 'messages': e}
    except openai.error.RateLimitError as e:
        print(f'OpenAI API request exceeded rate limit: {e}')
        return {'status': 'FAIL', 'messages': '앗 문제가 생겼어요! 다시 질문 해 주세요'}
    
def findKeyword(context,model="gpt-3.5-turbo",messages=[]):
    keywordList = []
    try:
      messages.append({'role': 'system', 'content': 'You are a sentence analyst.You have to answer in Korean.'})
      messages.append({'role':'user','content':"Please make shape of list What are the important noun words in a given sentence show only words\n context:"+context})
      #messages.append({'role':'user','content':context})
      response = openai.ChatCompletion.create(
          model = model,
          messages = messages,
          temperature = 0
      )
      context = response.choices[0].message.content
      keywordList = str(context).split(',')
      print(context)
      findKeywordFromDB(keywordList)
    except openai.error.APIError as e:
         print(f'OpenAI API returned an API Error: {e}')
         return {'status':'FAIL','messages':e}
    except openai.error.APIConnectionError as e:
        print(f'Failed to connect to OpenAI API: {e}')
        return {'status': 'FAIL', 'messages': e}
    except openai.error.InvalidRequestError as e:
        print(f'Invalid Request to OpenAI API: {e}')
        return {'status': 'FAIL', 'messages': e}
    except openai.error.RateLimitError as e:
        print(f'OpenAI API request exceeded rate limit: {e}')
        return {'status': 'FAIL', 'messages': '앗 문제가 생겼어요! 다시 질문 해 주세요'}

def findKeywordFromDB(keyword):
    connect = cx_Oracle.connect("PROJECT", "PROJECT", "localhost:1521/xe")
    for key in keyword:
        df=pd.read_sql_query(
                  f""" 
                          SELECT PRODUCTNAME ,NO
                          FROM FOODTABLE WHERE PRODUCTNAME LIKE '{key}' """ 
                          , con = connect)
        connect.close()
    

if __name__ == '__main__':
    try:
        message=[]
        print('안녕하세요 큐봇입니다! 무엇을 도와드릴까요?')
        while True:
            content = input('')
            response = chatbot(content,messages=message)
            if response['status'] == 'SUCCESS':
                messages = response['messages']
                answer = response['messages'][len(messages)-1]['content']
                print(f'큐봇:{answer}')
            else:
                print(response['messages'])
                break
            #print(response)
    except KeyboardInterrupt as e: #ctrl+c
        print('종료합니다')