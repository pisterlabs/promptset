import openai
import os
import pandas as pd
import json


testKey = os.environ.get("API_KEY")
openai.api_key = testKey


file = open("response.txt","r")
s:str = file.read()
j = json.loads(s)
#print(j)

#prompt = "You will expand upon the ideas given for an educational game by providing more vivid and thorough descriptions for each idea"
outputFile = open("expandedGames.txt", "w")
def askDescription(x):
    response = openai.ChatCompletion.create(
          model='gpt-4',
          messages=[
            # {"role": "system", "content": "You are an educational mini-game generator."},
            {"role": "system", "content": "You are tasked to take a description of an educational game and expand upon it by creating characters and dialoge, describing the setting, and actions done by the user and character "},
            {"role" : "user", "content" : "Please expand on the following game description: " + str(x)}
          ]
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def askMechanics(d, desc):
    response = openai.ChatCompletion.create(
          model='gpt-4',
          messages=[
            # {"role": "system", "content": "You are an educational mini-game generator."},
            {"role": "system", "content": "You are tasked on making the gameplay mechanics of the given gameplay description and general outline of the mechanics. Make sure what the output can be easily read and programmed"},
            {"role" : "user", "content" : "The description of the game is " + desc[:8000]}, #lmits to first 8000 characters
            {"role" : "user", "content" : "Please expand on the following: " + str(d)}
          ]
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def askFlow(d, desc, mech):
    response = openai.ChatCompletion.create(
          model='gpt-4',
          messages=[
            # {"role": "system", "content": "You are an educational mini-game generator."},
            {"role": "system", "content": "You are tasked to make the general flow of the given game description, game mechanics, and flow description. please be sure identify any levels that are in the description and expand upon how a user would play through the game. Also identify which settings and characters are in each level"},
            {"role" : "user", "content" : "The description of the game is " + desc[:8000]},
            {"role" : "user", "content" : "The mechanics of the game is " + mech[:8000]},
            {"role" : "user", "content" : "Please expand on the following: " + str(d)}
          ]
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


for game in j:
    desc = ""
    mech = ""
    flow = ""
    outputFile.write("original game: " + str(game))
    for d in game:
        print(d, ' : ' , game[d])
        if ("goal" in str(d).lower() or "code" in str(d).lower()):
            continue
        elif ("description" in str(d).lower() ):
            desc = askDescription(game[d])
        elif ("mechanic" in str(d).lower()):
            mech = askMechanics(game[d],desc)
        elif ("flow" in str(d).lower() ):
            flow = askFlow(game[d],desc,mech)
        #breakpoint()
    outputFile.write("description: "+ desc)
    outputFile.write("mechanics: "+ mech)
    outputFile.write("game flow: "+ flow)

# response = openai.ChatCompletion.create(
#   model='gpt-4',
#   messages=[
#     # {"role": "system", "content": "You are an educational mini-game generator."},
#     {"role": "system", "content": prompt},
#     {"role": "user", "content": learningGoals},
#     {"role": "user", "content": subGoals},
#     {"role" : "user", "content" : "Please format the learning goal code, description, game mechanics, game flow, and the sub-goals to be on seperate lines, each minigame should start with MINIGAME"}
#   ]
# )