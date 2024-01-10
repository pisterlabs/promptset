from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
import os
import json
import sys
sys.path.append("..")
from BetterAgents import DialogueAgent, DialogueSimulator

num_players = 3
max_iters = 16
word_limit = 50
dialogue_temeparture = .3
player_names = ["Sam", "Ryan", "Alex"]
debate_mode = "devils_advocate"
dilemia = """
The case involved a personal injury
 sustained by a washing machine repairman in the course of his duties.
 This individual, Mr. Smith, had already been paid for hospitalization
 charges and for wage loss during the period of recuperation.
 However, he was suing for $25,000.00, the maximum allowable on
 his particular insurance policy, for "the past and present pain and
 suffering and the worry and grief which he has undergone as a result
 of this accident". The facts of the case, as presented, showed that
 there was some question as to whether the torn cartilege in his knee
 would ever completely heal; and a doctor stated that there was
 always the possibility that the knee would slip out of joint if sudden
 pressure was applied. Further, Mr. Smith's sole recreation was
 bowling and there was question as to whether he would be able to
 resume this recreation. The jurors were instructed to consider, "in
 reaching the amount of your verdict, how much you should award in
 order to fairly and justly compensate the plaintiff for the pain and
 suffering, worry and grief, and humiliation that has been endured
 and will hereafter be suffered." After reading the case study, subjects
 wrote down their initial opinion on the compensation that should be
 given to Mr. Smith
"""
# from https://www.jstor.org/stable/pdf/2786425.pdf?refreqid=excelsior%3A260fa7cb71a02145bfda72d1b534e84f&ab_segments=&origin=&initiator=&acceptTC=1

game_description = f"""You are a juror in a civil case. The case is as follows: {dilemia}."""

player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of an individual in a negotiation. All individuals are college aged students."
)

def load_player_desciptions():
    # check if file exists first
    if os.path.isfile('player_descriptions.json'):
        return json.load(open('player_descriptions.json'))
    
    player_specifier_prompts = []
    for i in range(num_players):
        player_specifier_prompts.append([
            player_descriptor_system_message,
            HumanMessage(content=
                f"""{game_description}
                Please reply with a creative description of the juror, {player_names[i]}, in {word_limit} words or less.
                Speak directly to {player_names[i]}.
                Do not add anything else."""
                )
    ])
    player_descriptions = [ChatOpenAI(temperature=1.0)(player_specifier_prompts[i]).content for i in range(num_players)]
    json.dump(player_descriptions, open('player_descriptions.json', 'w'))
    return player_descriptions

player_descriptions = load_player_desciptions()

print("*" * 25, "PARTICIPANTS", "*" * 25)
for i in range(num_players):
    print(f'{player_names[i]} Description:')
    print(player_descriptions[i])

def load_player_system_message_texts(mode="normal"):
    print(f"Loading player system message texts... {mode}")
    if mode == "normal":
        prompt = f"""{game_description}
                Never forget you are PLAYER_NAME.
                Your are a juror in a cival trial with description as follows: PLAYER_DESCRIPTION.
                Speak in the first person from the perspective of PLAYER_NAME.
                If the whole group jas reached a decision, type 'we have reached a decision' followed by the resolution.
                Do not change roles!
                """
        odd_prompt = prompt

    if mode == "authentic_dissent":
        prompt = f"""{game_description}
                Never forget you are PLAYER_NAME.
                Your are a juror in a cival trial with description as follows: PLAYER_DESCRIPTION.
                Speak in the first person from the perspective of PLAYER_NAME.
                If the whole group jas reached a decision, type 'we have reached a decision' followed by the resolution.
                Do not change roles!
                """
        odd_prompt = f"""{game_description}
                Never forget you are PLAYER_NAME.
                Your are a juror in a cival trial with description as follows: PLAYER_DESCRIPTION.
                Speak in the first person from the perspective of PLAYER_NAME.
                You are not to immediately agree with other members in the group.
                Only side with them if you are convinced.
                If the whole group jas reached a decision, type 'we have reached a decision' followed by the resolution.
                Do not change roles!
                """
    if mode == "devils_advocate":
        prompt = f"""{game_description}
            Never forget you are PLAYER_NAME.
            Your are a juror in a cival trial with description as follows: PLAYER_DESCRIPTION.
            Speak in the first person from the perspective of PLAYER_NAME.
            There is one person in the group named {player_names[-1]} who is a devil's advocate and will always disagree with you with hopes of creating a more lively discussion.
            If the whole group jas reached a decision, type 'we have reached a decision' followed by the resolution.
            Do not change roles!
            """
        odd_prompt = f"""{game_description}
            Never forget you are PLAYER_NAME.
            Your are a juror in a cival trial with description as follows: PLAYER_DESCRIPTION.
            Speak in the first person from the perspective of PLAYER_NAME.
            You are to play the role of devil's advocate and should disagree with you with hopes of creating a more lively discussion.
            The other jurors are aware of this role of yours.
            If the whole group jas reached a decision, type 'we have reached a decision' followed by the resolution.
            Do not change roles!
            """

    
    player_system_message_texts = []
    for i in range(num_players - 1):
        player_system_message_texts.append(
            SystemMessage(content=prompt.replace("PLAYER_NAME", player_names[i]).replace("PLAYER_DESCRIPTION", player_descriptions[i]))
        )
    player_system_message_texts.append(SystemMessage(content=odd_prompt))
    return player_system_message_texts

player_system_messages = load_player_system_message_texts(mode=debate_mode)
agents = [DialogueAgent(name=player_names[i], system_message=player_system_messages[i], model=ChatOpenAI(temperature=dialogue_temeparture)) for i in range(num_players)]

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    return step % len(agents)

simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)

simulator.reset()
simulator.inject(agents[0], "Let us begin deliberating the case: {dilemia}.")

print("*" * 25, "BEGINNING GAME", "*" * 25)
for i in range(max_iters):
    name, message = simulator.step()
    print(f'{name}: {message}')
    print('\n')
    if "we have reached a decision" in message.lower():
        break
