import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

def gpt4_warning(sms, retry = 10):
    #TODO: Do ten tries max
    gotAnswer = False
    while gotAnswer == False and retry > 0: 
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a system that warns users about the dangers of scams."
                        + "The following message has been marked as a possible scam. Give the user a warning"
                        + "and whatever information might help them understand why this might be a scam.",
                    },
                    {"role": "user", "content": sms},
                ],
                max_tokens=100
            )
            gotAnswer = True
            
            answer = response["choices"][0]["message"]["content"].strip()

            return answer
        except:
            print("Error, trying again") 
            retry -= 1
            time.sleep(0.005)
    return "⚠️ WARNING: The message you received may be a potential scam! ⚠️"