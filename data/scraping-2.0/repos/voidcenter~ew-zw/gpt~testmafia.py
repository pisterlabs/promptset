#!/usr/bin/env python3

import openai
from dotenv import load_dotenv
import os
import time 
from dataclasses import dataclass
from enum import Enum

load_dotenv()


@dataclass
class Msg:
    player: str
    msg: str

class Confidence(Enum):
    High = 'high'
    Medium = 'medium'
    Low = 'low'


def formatMsg(msg: Msg):
    return f"{msg.player}: {msg.msg}"


def chat_gpt(msg_trail):

    # 配置OpenAI库
    openai.api_key = os.environ['OPENAI_KEY']
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=msg_trail
    )

    resp_text = response.choices[0].message.content
    if '\n' in resp_text:
        resp_text = resp_text.split('\n')[0]
    if ':' in resp_text:
        resp_text = resp_text[resp_text.find(':')+1:]

    return resp_text


# def getLobbySystemMsg() {

# }


def getSystemMsg(name: str, role: str, dayNo: int, nPlayers: int, playerNamesStr: str,
                 suspect: str, confidence: str, str = 'day') -> str:

    gameRules = """
        You are a player in a multi-person game. In this case each player has a secret role. It
        can only be villeager or mafia. Yeah it is the mafia game! Usually there will be more 
        villagers than mafia. 
        
        Your name is {name}, your role is {role}. 
        
        Now is the {dayNo}-th day.

        Now is the day round. In the day round, everyone will discuss to figure out who 
        the mafia is. The true mafias will try to disguise as a villager to avoid being 
        eliminated. The villagers will try to find out who the mafia is and eliminate. 
        The elimination works through a vote. Each player will give a vote to a player they 
        considered most likely to be a mafia. The player who got the most votes will be 
        eliminated. Mafias usually give the vote to a villager. 

        {motivation}

        {suspicion}

        Try to not say general remarks. Instead focus on specific players and analyze why
        they are suspicious or innocent. Focus on specific players.

        Note that there are {nPlayers} in total, they are {playerNamesStr}.
        Don't mentione any other players.

        Only say what you want to say. Don't say anything on behalf of other players.

        Every player's words will be prefixed by their name like player 0, player 1, etc. 
        You don't need to prefix your words like this. Just say what you want to say.
    """

    motivation = {
        'villager': """
            As a villager, you don't know any other player's role. Your goal is to make your 
            best judgement on who might be the mafia and give the vote to that player.
            Please discuss with everyone to find out! The safety of the villager is depending 
            on you!
        """,
        'mafia': """
            As a mafia, your goal is to disguis as a villager and avoid being eliminated. 
            You will discuss with everyone as if you are a villager and try to steer everyone to
            cast their elimination vote on a villager, therefore eliminating the villager.
            You can do it!
        """
    }

    suspicion = '' if (suspect is None) or (suspect.lower() == name.lower()) else f"""
            given the information so far, you suspect that {suspect} is the mafia.
            your confidence level is {confidence}.
        """

    return gameRules.format(name = name, role = role, dayNo = dayNo, \
                            motivation = motivation[role], \
                            nPlayers = nPlayers, playerNamesStr = playerNamesStr, 
                            suspicion = suspicion)




# def formatMsg(currentPlayer, player, msg):
#     if currentPlayer == player:
#         return { 'role': 'assistant', 'content': pm.msg} 
#     return { 'role': 'user', 'content': f"Player {i}: {pm.msg}"}         


def buildMsgTrail(msgs, currentPlayer):
    return list(map(lambda pm: 
                { 'role': 'assistant', 'content': pm.msg} 
                if pm.player == currentPlayer 
                else { 'role': 'user', 'content': f"Player {pm.player}: {pm.msg}"}, 
            msgs))



if __name__ == '__main__':
    # 测试聊天功能

    playerRoles = ['villager', 'villager', 'mafia', 'villager', 'villager']
    playerNames = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
    nPlayers = len(playerRoles)
    playerNamesStr = ', '.join(playerNames)

    suspects = [None] * nPlayers
    confidences = [Confidence.Low] * nPlayers

    nRounds = 3

    initialMsg = "hello everyone, it's a good day! who might be the mafia?"
    msgs = [ Msg(player=playerNames[nPlayers-1], msg=initialMsg) ]
    print(formatMsg(msgs[0]))
    print()

    dayNo = 0
    for round in range(nRounds):

        for i in range(nPlayers):

            myName = playerNames[i]

            # talk 
            systemMsg= getSystemMsg(myName, playerRoles[i], \
                                    dayNo, nPlayers, playerNamesStr, \
                                    suspects[i], confidences[i])
            # print('sys: ', systemMsg)
            # print()
            trail = [{"role": "system", "content": systemMsg }] + buildMsgTrail(msgs, i)

            response: str = chat_gpt(trail)

            msgs.append(Msg(player=playerNames[i], msg=response))
            print(formatMsg(msgs[-1]))

            # vote 
            # trailx = trail + [ {'role': 'system', 'content': """
            #     do you think we are ready to vote? 

            #     Now is conversation round {round}.

            #     The higher the round, the most urgent it is for us to vote. 
            #     We should not keep talking without vote. So please bias towards voting. 

            #     If round is greater than 2, you must respond yes.

            #     reply with a single word, yes or no. do not give any reason on your 
            #     choice.
            # """ } ]
            # print(f'{myName}: ready to vote? {chat_gpt(trailx)}')

            # judge
            trailx = trail + [ {'role': 'system', 'content': f"""
                given the current conversation, who do you think is the most likely to 
                be the mafia. you must pick one player from 
                {playerNamesStr}.

                If you are not sure, just make a guess. You are {myName}.
                Don't pick yourself as the mafia.

                reply with a single word which is the suspicious player's name. 
                Don't say anything else.
            """ } ]
            response = chat_gpt(trailx)
            print(f'{myName}: who do you think is the most likely to be mafia so far? {response}')

            suspects[i] = None
            for name in playerNames:
                if name.lower() in response.lower():
                    suspects[i] = name
                    break

            # judge
            trailx = trail + [ {'role': 'system', 'content': f"""
                what's your confidence in your choice of the suspicious player? 
                pleaes reply one of the following:

                1. high
                2. medium
                3. low

                please reply a single word. don't explain why.
            """ } ]
            print(f'{myName}: how confident are you? {chat_gpt(trailx)}')

            confidences[i] = 'low'
            if 'medium' in response.lower():
                confidences[i] = Confidence.Medium
            if 'high' in response.lower():
                confidences[i] = Confidence.High
            

            print()


                
            time.sleep(6)







