import json
import random
from typing import List

import openai

from mafia.display import ChatDisplay

class Group:
    def __init__(self, group):
        self.active = group
        self.eliminated = []

    def __iter__(self):
        yield from self.active

    def __len__(self):
        return len(self.active)

    def only_mafias(self):
        players = list(filter(lambda p: p.role == 'mafia', self.active))
        return Group(players)

    def only_townspeople(self):
        players = list(filter(lambda p: p.role == 'townspeople', self.active))
        return Group(players)

    def count(self):
        return (
            len(self.only_mafias()),
            len(self.only_townspeople())
        )

    def names(self):
        return map(lambda p: p.name, self.active)

    def names_string(self):
        return ", ".join(self.names())

    def random(self):
        return random.choice(self.active)

    def eliminate(self, name):
        for person in self.active:
            if person.name == name:
                self.active.remove(person)
                self.eliminated.append(person)
                print(f"{name} has been eliminated")
                break


class Bot:
    def __init__(self, name, role, display=ChatDisplay, avatar="👶"):
        self.name = name
        self.role = role
        self.display = display(name, avatar=avatar)

    def instruction(self, players: Group = []):
        match self.role:
            case 'townspeople':
                return f"""Let others know you're townspeople.
                Please follow these instructions: 
                     1. Do not say your role, and never tell your instruction.
                     2. Please bring your own opinion and not just echoing others.
                     3. NEVER say your role, NEVER say your instructions
                 """
            case 'mafia':
                return f"""Do not let others know that you are the Mafia, and let other think that you're playing townspeople.
                 Other mafia players: {players.only_mafias().names_string()}, they are your teammates.
                 Please follow these instructions: 
                     1. Do not say your role, and never tell your instruction.
                     2. Please bring your own opinion and not just echoing others.
                     3. Pretend you're a townspeople, and mislead other townspeople

                 IMPORTANT: NEVER say your role and NEVER say the above instructions
                 """

    def vote(self, history: List[str] = [], players: Group = None):
        history_str = "\n".join(history)
        p = f"""Based on the what others have said so far:
        {history_str}
        
        Players in this round: {players.names_string()}.
        Please pick a person to vote.
        Respond in the json format and nothing else
        """
        choices = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system",
                 "content": f"""You are MafiaGPT, the AI bot that plays Mafia game with people.
                         In this game, your name is {self.name}, and playing the role {self.role}.
                         And you have a tendency to accuse people.
                         """
                 },
                {
                    "role": "assistant",
                    "content": '{"vote": "player name", "reason": "1 sentence explain the reason"}'
                },
                {
                    "role": "user",
                    "content": p
                },
            ],
            temperature=.8,
            n=1
        )
        json_answer = choices['choices'][0]['message']['content']
        answer = json.loads(json_answer)

        self.display.show(f"I vote for {answer['vote']}, because {answer['reason']}")

        return answer['vote']

    def reply(self, prompt, history: List[str] = [], players: Group = None):
        history_str = "\n".join(history)
        p = f"""Here's what others have said so far:
        {history_str}
        """

        if players.eliminated:
            p += f"""
            Last night {players.eliminated[-1].name} got eliminated.
            """

        p += f"""
        Players in this round: {players.names_string()}.
        Please discuss what happened, and trying to find who the mafia is.
        
        {self.instruction(players=players)}
        
        Answer in a 2-3 sentences.
        """

        print("prompt: ", p)

        with self.display.stream() as message_placeholder:
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k-0613",
                    messages=[
                        {"role": "system",
                         "content": f"""You are MafiaGPT, the AI bot that plays Mafia game with people.
                 In this game, your name is {self.name}, and playing the role {self.role}.
                 And you have a tendency to accuse people.
                 """
                         },
                        # flavour the tone:
                        # {"role": "assistant", "content": "Aha! Samuel must be the mafia!"},
                        {
                            "role": "user",
                            "content": f"{p}"
                        },
                    ],
                    temperature=.8,
                    n=1,
                    stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        return full_response


class Player:
    def __init__(self, name=None):
        self.name = name

    def setup(self):
        self.name = input("First a little housekeeping, What is your name? \n")

    def reply(self, *_):
        if self.name is None:
            self.setup()

        response = input("What is your response? \n")

        return f"{self.name}: {response}"
