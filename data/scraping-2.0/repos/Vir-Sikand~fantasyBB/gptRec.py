from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI()


def getRec(player_list) :
    toPass = createDocument(player_list)
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a professional basketball analyst, skilled in explaining fantasy basketball predictions and weighing their pros and cons."},
        {"role": "user", "content": "From this list of players and their attributes, choose the player you think would be best to add to a fantasy basketball roster based on their current season stats and give three reasons why in three short sentences: " + str(toPass)}
    ]
    )
    return (completion.choices[0].message)


def createDocument(player_list) :
    arr = []
    for p in player_list :
        player = p[1]
        name = " name:" + player.name
        injuryStatus = " injuryStatus: " + player.injuryStatus
        injured = " injured: " + str(player.injured)
        #stats (Will implement later when I figure out how to only get recent performance)
        total_points = " total points: " + str(player.total_points)
        avg_points = " avg points: " + str(player.avg_points)
        projectedTotal = " projected total points: " + str(player.projected_total_points)
        projectedAvg = " projected average points: " + str(player.projected_avg_points)
        s = name + injuryStatus + injured + total_points + avg_points + projectedTotal + projectedAvg
        arr.append(s)
    return arr


