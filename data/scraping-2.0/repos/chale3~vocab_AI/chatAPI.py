import openai
import time


def w25(message, flag = 0):
    flag += 1
    try:
        openai.api_key = open("keys/key3.txt", "r").read().strip("\n")
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            temperature=0.2,
            max_tokens=1000,
            frequency_penalty=0.0
        )
        reply_content = completion.choices[0].message.content
        return(reply_content)
    except:
        while flag <= 5:
            print("Retrying... " + str(flag) + " of 5")
            time.sleep(1)
            return w25(message, flag)
        return("Error: Chat API failed to respond.")
    
