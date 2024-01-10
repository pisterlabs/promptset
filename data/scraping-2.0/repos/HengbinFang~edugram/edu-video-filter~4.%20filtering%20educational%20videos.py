import requests
import openai
import math


openai.api_key = ""
session = requests.Session()

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def build(desc, transcription):
    return deEmojify(f"{desc} | {transcription}  ->")

def logprobs_to_probs(logprobs):
    probs = []
    for x in logprobs:
        probs.append(math.exp(x))
    return probs


def isEducational(desc, transcription):
    response = openai.Completion.create(
          model="ft:babbage-002:personal::8Tvt6kYy",
          prompt=build(
            desc=desc,
            transcription=transcription
          ),
          temperature=1,
          top_p=1,
          logprobs=25, # see probabilities of all of em
          frequency_penalty=0,
          presence_penalty=0,
          stop=["."]
        )
    logprobs = list(response["choices"][0]["logprobs"]["top_logprobs"][0].values())
    print(logprobs)
    t1,t2=logprobs[0],logprobs[1]

    is_true = "true" in response["choices"][0]["text"].lower()
    return is_true, logprobs_to_probs([t1, t2])

def reel_data(id):
    if not id:
        return {}
    data = session.get(f"https://www.instagram.com/api/v1/oembed/?hidecaption=0&maxwidth=540&url=https://www.instagram.com/reel/{id}").json()
    return data

with open("english_videos.txt", "r", encoding="utf-8") as file:
    df = file.read().splitlines()

with open("educational_videos.txt", "a", encoding="utf-8") as file:
    for x in df:
        id, text = x.split("|||")
        try:
            d = reel_data(id)
            if d:
                desc = d["title"]
                edu = isEducational(desc, text)
                if edu[0]:
                    print('Educational!!! Writing', id, edu[1])
                    file.write(f"{id}\n")
                else:
                    print("Not educational", id, edu)
        except Exception as er:
            print("Error:", er)

