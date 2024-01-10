import openai
import json
import asyncio

OPENAI_API_KEY = "openai-api-key"
openai.api_key = OPENAI_API_KEY


# 텍스트 기반 감정 분석 with chatgpt
async def chat_emotion(who, user_content):
    if who == "gpt":
        emotion = "happiness, sadness, anger"
        emotion_dict = {
            "emotion": {
                "happiness": 0.9,
                "sadness": 0.0,
                "anger": 0.0,
            }
        }
    else:
        emotion = "happiness, excited, sadness, bored, disgust, anger, calm, comfortable"
        emotion_dict = {
            "emotion": {
                "happiness": 0.9,
                "sadness": 0.0,
                "anger": 0.0,
                "excited": 0.0,
                "bored": 0.0,
                "disgust": 0.0,
                "calm": 0.0,
                "comfortable": 0.0,
            }
        }
    # 감정 분석 Prompt
    system_content = f"""
        I want you to help me to do text-based emotion analysis. 

        Please analyze its emotion and express it with numbers.
        emotion(rates of {emotion})

        Provide them in JSON format with the following keys: emotion

        Examples:
        {emotion_dict}

        Also, you should observe the format given in the example. 
        Don't add your comments, but answer right away.
    """
    # ChatGPT 감정 분석
    messages = []
    result = None
    messages.append({"role": "system", "content": f"{system_content}"})
    messages.append({"role": "user", "content": f"{user_content}"})
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        assistant_content = completion.choices[0].message["content"].strip()
        result = json.loads(assistant_content.replace("'", '"'))
    # 감정을 알수 없을경우 happy로 지정
    except Exception as e:
        print("emotion err : 감정을 확인할 수 없습니다")
        result = emotion_dict
    finally:
        print(who + " : ", result)
    return result
