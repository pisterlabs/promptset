
import os
import openai
import json
import time
from dotenv import load_dotenv

dinos = ["티라노사우루스가",
"트리케라톱스가",
"벨로시랩터가",
"파라사우롤로푸스가",
"이구아노돈이",
"프테라노돈이",
"안킬로사우루스가",
"아르헨티노사우루스가",
"파키케팔로사우루스가",
"엘라스모사우루스가",
"모사사우루스가",
"오우라노사우루스가",
"인시시보사우루스가",
"람베오사우루스가",
"노도사우루스가",
"오비랍토르가",
"케찰코아틀루스가",
"켄트로사우루스가",
"스피노사우루스가",
"스테고사우르스가",
"브라키오사우루스가",
"알로사우루스가",
"딜로포사우루스가",
"다센트루루스가",
"친타오사우루스가",
"콤프소그나투스가",
"기가노토사우루스가", 
"디모르포돈이",
"사우로파가낙스가",
"카스모사우루스가"]
topics = [
    "아름다운","재밌는","흥미로운"
]

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
    
def chat(dino,topic):

    stime = time.time()
    print(dino+':start')
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content":"You are smart storyteller"},
            {"role": "user", "content": dino+" 주인공인 "+topic+" 동화를 아이들에게 짧게 들려줘 "},
        ],
    )
    etime = time.time()
    print("time:",etime-stime)
    return completion

stories = []
for topic in topics:
    for dino in dinos:
        completion = chat(dino,topic)
        story = {
            "name" : dino[:-1],
            "story" : completion.choices[0].message.content
        }
        stories.append(story)
        pass
    
story_json = json.dumps(stories)
f = open("stories.json","w")
f.write(story_json)
f.close()
print(stories)