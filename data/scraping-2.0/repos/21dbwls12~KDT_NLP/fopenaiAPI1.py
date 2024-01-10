from openai import OpenAI
import os
# 루트디렉토리에 .env 파일을 만들고 아래와 같이 OPENAI_API_KEY와 TOUR_ASSISTANT_ID를 입력 후 해당 변수들을 현재 파일로 로드
from dotenv import load_dotenv

def qna(question) :
    load_dotenv()

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    tour_assistant_id = os.getenv('TOUR_ASSISTANT_ID')
    TOUR_ASSISTANT_ID = tour_assistant_id

    messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
    messages.append(
                {"role": "user", "content": question},
            )
    
    chat = client.chat.completions.create(
    model="gpt-4-1106-preview", messages=messages )
    
    return chat.choices[0].message.content
        
def getImage(question) :
    load_dotenv()

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
    # messages.append(
    #             {"role": "user", "content": question},
    #         )
    
    getImage = client.images.generate(
        model="dall-e-3",
        prompt= question,
        n= 1,
        size= "1024x1024",
        # messages = messages,
    )
    # model="gpt-3.5-turbo", messages=messages )
    
    # if getImage.data[0].url == None :
    #     return None
    # else :
    return getImage.data[0].url
    # return chat.dict()["data"][0]["resource"]
    # return chat.choices[0].message.content
