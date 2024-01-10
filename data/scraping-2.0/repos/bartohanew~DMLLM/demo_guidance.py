import guidance
from guidance import assistant, system, user, gen
from dotenv import load_dotenv
import json
load_dotenv()

dm = guidance.models.OpenAIChat("gpt-4-1106-preview")
#dm = guidance.models.OpenAIChat("gpt-3.5-turbo")

commands_description = """
Here are the available commands:
    + ask_assistant: <your question>
        Whenever you are unsure of how to properly execute your duties, you may ask your assistant for help.ond "ask_assistant: <your question>".
    + roll_die: <number of sides>, <number of dice>, <type>
        When a player attempts an action that has a chance of failure.
        Type can be any of the following: strength, dexterity, constitution, intelligence, wisdom, charisma, attack.
    + spell_lookup: <spell name>
        When a player attempts to cast a spell, you may respond with "spell_lookup: <spell name>".
    + confusion: <problem with executing the action>
        When you cannot fully understand what to do, use this command.
        You may end the session without any outcomes as long as you state why you are confused.
    + objection: <player's action>
        When a player attempts an action that is not allowed by the rules, you may respond with "objection: <player's action>".
    + outcome: <change in the world>
        In order to specify a change in the world, you may respond with "outcome: <change in the world>".
    + END
        When you are finished considering the implications, you may respond with "END".
"""

with system():
    dm += "You are the rulemaster for a game of Dungeons and Dragons."

    dm += """
    The current state of the world is:
    + There are three players: Alice, Bob, and Charlie.
    + Alice
        + 10 hit points
        + human wizard
        + can cast fireball, magic missile, and shield
        + has a dagger
        + has a spellbook
    + Bob
        + 10 hit points
        + human fighter
        + has a sword
    + Charlie
        + 10 hit points
        + human cleric
        + can cast cure wounds
        + has a mace
    + There are three monsters: a goblin, an orc, and a troll.
    + goblin
        + 3 hit points
        + 30 ft from Alice, 100ft from Bob, 50ft from Charlie
    + orc
        + 6 hit points
        + 50 ft from Alice, 30ft from Bob, 100ft from Charlie
    + troll
        + 10 hit points
        + 100 ft from Alice, 50ft from Bob, 30ft from Charlie
    """

with user():
    dm += "Alice: I cast fireball at the goblin."

modifiers = {
    "strength": 4,
    "dexterity": -1,
    "constitution": -2,
    "intelligence": 0,
    "wisdom": 2,
    "charisma": 0,
}


def spell_lookup(name):
    name = name.strip().lower()

    print('searching for spell', name)

    import requests
    nameq = name.strip().replace(' ', '+')
    url = f"https://www.dnd5eapi.co/api/spells/?name={nameq}"
    response = requests.get(url)
    response = response.json()['results']

    response = [r for r in response if r['name'].lower() == name.lower()]

    full = []
    for result in response:
        url = result['url']
        response = requests.get(f"https://www.dnd5eapi.co{url}")
        full.append(response.json())

    if not len(full):
        return "No spells found."
    
    result = []
    for f in full:
        result.append(f"{f['name']}\n\tDescription: {f['desc'][0]}\n\tRange: {f['range']}\n")

    return "\n".join(result)

objections = []
outcomes = []

while True:
    dmp = dm.copy()

    with assistant():
        dmp += "Let me describe very briefly (one sentence, ideally) what to do next...\n"

    print(dmp)

    with assistant():
        dmp = dmp + gen("thought", max_tokens=200)
        print('THOUGHT', dmp["thought"])
        dm += "Thinking to myself... " + dmp["thought"]

    with system():
        dmp += "\n" + commands_description
        dmp += "\nThe only available commands are 'roll_die', 'spell_lookup', 'objection', 'outcome', and 'END'."
        dmp += "\nWhen this specific turn in D&D combat is over, respond with 'END' to move to the next player's turn."
        dmp += "\nAlice is the only player acting now. Do not act for others."

    with assistant():
        dmp += "\nI will now specify my command in the format '<command name>: <arguments>'.\n"
        dmp = dmp + gen("command")

    print('COMMAND', dmp["command"])
    c, *args = dmp["command"].split(":")
    args = ":".join(args).strip()

    with assistant():
        if c == "roll_die":
            sides, num, typ = args.split(",")
            import random
            n_sides = int(sides.strip())
            n_dice = int(num.strip())

            typ = typ.strip().lower()
            result = [ random.randint(1, n_sides) for _ in range(n_dice) ]
            result = sum(result)

            mod = modifiers[typ] if typ in modifiers else 0

            self_message = f"I rolled a {result} + {mod} = {result+mod} for {typ}."

        elif c == "spell_lookup":
            self_message = "I looked up the spell '%s', finding the following:\n"%args + spell_lookup(args)

        elif c == "objection":
            self_message = "I make the objection '%s'" % args
            objections.append(args)

        elif c == "outcome":
            self_message = "I specify the outcome '%s'" % args
            outcomes.append(args)

        elif c == "END":
            break

        else:
            self_message = "I gave an invalid command, '%s'" % dmp["command"]

        print("ACTION RESULT:", self_message)
        dm += "Action completed: " + self_message + "\n"

print("Objections:")
for o in objections:
    print('+ ', o)

print("Outcomes:")
for o in outcomes:
    print('+ ', o)