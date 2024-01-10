import openai
import os
import time
#https://platform.openai.com/docs/guides/gpt/chat-completions-api의 질의어로 테스트해보자
openai.api_key = os.getenv('OPENAI_API_KEY')

def chatbot(content,model="gpt-3.5-turbo",messages=[],temperature=0):
    try:
        messages.append({'role':'user','content':content})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        answer = response.choices[0].message.content
        messages.append({'role':'assistant','content':answer})
        return {'status':'SUCCESS','messages':messages}


    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        return {'status':'FAIL','messages':e}
    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        return {'status':'FAIL','messages':e}
    except openai.error.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        return {'status':'FAIL','messages':e}
    except openai.error.InvalidRequestError as e:
        print(f'Invalid Request to OpenAI API:{e}')
        return {'status': 'FAIL', 'messages': e}

if __name__ == '__main__':
    try:
        messages=[]
        while True:
            content = input('질의하세요:')
            response= chatbot(content,messages=messages)
            #print(response)

            messages = response['messages']
            if response['status'] =='SUCCESS':
                answer = response['messages'][len(messages)-1]['content']
                print(f'코봇:{answer}')
            else:
                print(messages)
                break
    except KeyboardInterrupt as e:#CTRL+C입력시
        print('종료합니다')
