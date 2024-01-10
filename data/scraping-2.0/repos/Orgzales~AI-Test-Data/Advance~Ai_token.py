import openai
import os

#Ai token : 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

all_messgaes = [
        {
            "role": "system",
            "content": "You are my Dungeon Master for Dungeons & Dragons 5e."
        }
        # {
        #     "role": "user",
        #     "content": "Tell me a short story."
        #     # "content": "I'm a Level 1 Charater that's a rouge class and the race human. Please begin a Campiagne for me and continue it giving me "
        #     #            "options what I would like to do."
        # }
    ]

openai.api_key = os.getenv("OPENAI_API_KEY")

def GPT_Response():
    global all_messgaes
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = all_messgaes,
        temperature = 1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    string = response.choices[-1]
    all_messgaes.append(string["message"])
    print(all_messgaes[-1])

def New_Response(user_input):
    global all_messgaes
    all_messgaes.append({
        "role": "user",
        "content": str(user_input)
    })
    # print(all_messgaes[-1])
    GPT_Response()

while(True):
    print("\nYour input: ")
    user = input("")
    New_Response(user)

#command - set OPEN_API_KEY=<your key>
#command python my.file







