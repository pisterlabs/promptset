import os
import openai
openai.api_key = "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"
from gptResponse import generate_gpt3_response
from requests import get
import random as rn
from uuid import uuid1

def voice_generation(answerr):
    from gtts import gTTS
    import os
    name = str(input())
    language = 'en'
    myobj = gTTS(text=answerr, lang=language, slow=False)
    myobj.save(name + ".mp3")
    os.system("mpg321 welcome.mp3")

def splitting(answerr):
    answerr.split('.')
    return answerr

final = []


def image_generation(generationtext):
    temp = []
    for i in range(4):
        openai.Model.list()
        response = openai.Image.create(
            prompt=generationtext,
            n=1,
            size="256x256"
            )
        image_url = response['data'][0]['url']
        temp.append(image_url)
    return temp


downloadFlag = False


gptResult = generate_gpt3_response()
final = image_generation(gptResult)



def downloadimg(url):
    response = get(url)
    df = str(uuid1())
    with open("./images/"+df+".png", "wb") as f:
        f.write(response.content)
    
    
for i in final:

    downloadimg(i)

downloadFlag = True

