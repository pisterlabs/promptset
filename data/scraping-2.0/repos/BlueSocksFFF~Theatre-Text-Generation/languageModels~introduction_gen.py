import os
import openai
from dotenv import load_dotenv
import json

load_dotenv()

GPT3_API_KEY = os.getenv("GPT3_API_KEY")

openai.api_key = GPT3_API_KEY 

class gpt3_5_introduction_generator:

  def __init__(self) -> None:
     pass

  def generate_introduction(self, input, context):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "system", "content": "You are the artificial intelligence part of an improv show at The Annoyance Theater in Chicago. The show is meant to incorporate artificial intelligence into a live improv show. You can poke fun at yourself and others."},
                {"role": "assistant", "content": "This is the log of what has already been said: {}".format(context)},
                {"role": "user", "content": input}
                ],
      temperature=0.8,
      max_tokens=500,
      frequency_penalty=0.5,
      presence_penalty=0.0
    )["choices"][0]["message"]["content"]
    return response
  

def basic_intro():
  with open('castBio.json', 'r') as f:
              castBioDict = json.load(f)
  castList = list(castBioDict.keys()) ##list of improvisers in the show

  log = []

  for improviser in castList: ## going through each improviser to introduce them
    bio = castBioDict[improviser] ## each member has a bio

    print(improviser)

    introduction_bot_module = gpt3_5_introduction_generator() 
    introduction = introduction_bot_module.generate_introduction(input= "Introduce the improviser {} to the crowd in just a few lines. Do not be repetitive with the introductions. Do not say 'ladies and gentlemen'. They are one of the cast members of this show. You know this about them: {}. End by asking them a question.".format(improviser, bio), context=log)

    log.append(introduction)
  
    print("AI: ", introduction)

    reply = input("reply to the machine")

    print(improviser,": ", reply)

    follow_up = introduction_bot_module.generate_introduction(input="{} responds with {}. Give a short, quippy response. Do not ask any more questions.".format(improviser, reply), context=log)

    log.append(reply)

    log.append(follow_up)

    print("AI: ", follow_up)

    input("press ENTER to continue")


def blind_line_intro(players):
    
    log = []
    
    introduction_bot_module = gpt3_5_introduction_generator() 

    introduction = introduction_bot_module.generate_introduction(input= "Introduce an improv game called 'blind line'. The people playing this game will be {} In this game, three improvisers will begin a scene, but at any point, someone in the audience may yell the word 'bananas' and you will generate a random line that one of the improvisers will have to incorporate into the scene.".format(players), context=log)

    log.append(introduction)

    print("AI: ", introduction)

    

    
    



      