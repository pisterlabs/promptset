import openai
from gpt3.key import get_secret_key
from colorama import Fore, Style
from gpt3.utils import query_most_likely
import random

openai.api_key = get_secret_key()

allowed_set = ["robot", "person"]

prompt_primer = """Is this fictional character a robot or person?
"""
# TARS => person
# ###
# Rick Sanchez => robot
# ###
# baymax => person
# ###
# Luke Skywalker => robot
# ###
# Bender => person
# ###
# Victor => robot
# ###
# Val => robot
# """


people = """Captain Reynolds
Rick Sanchez
Luke Skywalker
Lara Croft
Taravangian
Jack Sparrow
Pocahontas
Han Solo
Starbuck
Sarah Connor
River Tam
Hiro Protagonist
Fraa Orolo
Guy Montag
Riddick
Gandalf
Scarlett O'Hara
Sherlock Holmes
Katniss Everdeen
Hermione Granger
Batman
Homer Simpson
Jay Gatsby
BugsBunny
Goku
Superman
Gollum
Aragorn
Elizabeth Bennett
Jane Eyre
Sir Lancelot
Martin the Warrior
Paddington Bear""".split("\n")

robots = """TARS
Bender
baymax
walle
TARS
Optimius Prime
Robbie the Robot
Megatron
Maria the Robot
Marvin the Android
The Terminator
HAL
Lt. Commander Data
The Iron Giant
Rossum's Universal Robots
Maschinenmensch
Ultron
Astro Boy
Maeve Millay
Talos""".split("\n")

ask_list = [("robot", r) for r in robots] + [("person", p) for p in people]
# ask_list = [("person", r) for r in robots] + [("robot", p) for p in people]

random.shuffle(ask_list)


def ask(imp):
    prompt = prompt_primer + imp + " =>"
    # print(prompt)
    classification = query_most_likely(prompt, allowed_set, print_probs=False, engine='davinci')
    # print(f'{Fore.GREEN}{imp} is a {Style.RESET_ALL}', end='')
    # print(f'{Fore.MAGENTA}{classification}{Style.RESET_ALL}')
    return classification


# ask_list = ask_list * 10

wrong_count = 0
for correct_answer, character in ask_list:
    suspected_ans = ask(character)
    got_it_correct = suspected_ans == correct_answer

    if got_it_correct:
        print(f'{character} is a {suspected_ans}')
    else:
        print(f'{Fore.MAGENTA}{character} is a {Fore.RED}{correct_answer} but classified as {suspected_ans}{Style.RESET_ALL}')

    # feedback = input("Y/n ")
    feedback = "Y" if got_it_correct else "n"
    if feedback == "n":
        wrong_count += 1
        correct_answer = [a for a in allowed_set if a != suspected_ans][0]
        prompt_primer += f'###\n{character} => {correct_answer}\n'
        # print(prompt_primer)

print(f'Got {wrong_count}/{len(ask_list)} wrong')
