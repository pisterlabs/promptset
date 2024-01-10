# Items literally have zero effect yet.
# Neither have skills
# TODO: add proper items and skills, even if they only have effect within triggers!

import random
import os

import openai

from trigger_files.trigger import get_func

openai.api_key = os.getenv('OPENAI_API_KEY')


class ADVENTURE:
    # possibility to write custom adventures
    # possibly an intelligent AI adventure writer
    # use maps or something similar to move properly
    def __init__(self, name, locations, npcs, secrets, flags, trigger=None, items=None, starting_stage=None):
        self.name = name
        if locations is None:
            self.locations = []
        else:
            self.locations = locations
        self.npcs = npcs
        self.secrets = secrets
        if trigger is None:
            trigger = []
        self.trigger = trigger
        self.flags = flags
        if starting_stage is None:
            starting_stage = {'location': self.rand_loc(), 'npcs': self.rand_npcs()}
        self.starting_stage = starting_stage
        if items is not None:
            self.items = items
        else:
            self.items = []
        for i in locations + npcs:
            if i.start_items is not None:
                self.items += i.start_items
        # https://github.com/AdrianSI666/DnD_Bestiary-Spellbook-CT
        # https://github.com/opendnd/personae Can do relations between characters!
        # https://github.com/topics/npc-generator

    def rand_loc(self):
        locations = []
        for i in self.locations:
            if i.activation_flag.value:
                locations.append(i)
        return random.choice(locations)

    def rand_npcs(self):
        npcs = []
        for i in self.npcs:
            if i.activation_flag.value:
                npcs.append(i)

        # weighted random choice for how many npcs
        weights = [0.1, 0.15, 0.25]  # proper distribution shifted to the left and centered around 3 would be better
        for i in range(len(npcs) - 3):
            weights.append(1 / (2 ** (i + 2)))
        k = random.choices(range(len(npcs)), weights=weights[:len(npcs)])[0]

        # weighted random choice which npc to choose
        weights = []
        for i in npcs:
            weights.append(i.chance_to_appear)
        npcs = random.choices(npcs, weights=weights, k=k)
        return npcs


class LOCATION:
    def __init__(self, name, activation_flag, secrets, description=None, start_items=None):
        self.name = name
        self.activation_flag = activation_flag
        self.description = description
        self.start_items = start_items
        self.secrets = secrets
        self.former_inputs = []
        self.former_outputs = []

    def __call__(self, message, npcs):
        npc_text = ''
        for i in npcs:
            npc_text += f'{i.name} is in the room.\n{i.description}\n'
        secret_text = ''
        if self.secrets is not None and self.secrets != []:
            secret_text = ''
            for i in self.secrets:
                secret_text += f'Secret: {i.prompt}\n'  # i.prompt[self] to adjust secret description
            secret_text = f'If somebody looks around closely they might find hints on some of these secrets:' \
                          f'\n{secret_text}\n:'
        prompt = f'{self.description}\n\n{secret_text}{npc_text}\nDescribe {self.name} ' \
                 f'and answer questions about {self.name}.\n\nQuestion: '
        for i in range(min(len(self.former_inputs), 3)):
            prompt += f'{self.former_inputs[i]}\nAnswer: {self.former_outputs[i]}\n\nQuestion: '
        prompt += f'{message}\n\nAnswer: '
        response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0.4,
                                            max_tokens=100, stop='Question:')
        response = response['choices'][0]['text']
        self.former_inputs.append(message)
        self.former_outputs.append(response)
        return response


class NPC:
    # Little idea on the side: giving each NPC an archetype from which it can generate a description and co.
    def __init__(self, name, activation_flag, secrets, description=None, start_items=None, skills=None,
                 chance_to_appear=1):
        self.name = name
        self.activation_flag = activation_flag
        self.description = description
        self.start_items = start_items
        self.skills = skills
        self.secrets = secrets
        self.chance_to_appear = chance_to_appear
        self.prompt = f'The following is a conversation between {self.name} and a person. {self.description} '
        self.former_speak_inputs = []
        self.former_speak_outputs = []
        self.former_describe_inputs = []
        self.former_describe_outputs = []

    def speak(self, message):
        # check prompt propperly
        if self.secrets is None or self.secrets == []:
            prompt = self.prompt + '\n\nPerson: '
        else:
            secret_text = ''
            for i in self.secrets:
                secret_text += f'Secret: {i.prompt}\n'  # i.prompt[self] to adjust secret description
            prompt = f'{self.prompt}{self.name} knows some secrets and loves to talk about them.' \
                     f'\n{secret_text}\nPerson: '
            # The single \n before person is correct.
        for i in range(min(len(self.former_speak_inputs), 3)):
            prompt = f'{prompt}{self.former_speak_inputs[i]}\n{self.name}: {self.former_speak_outputs[i]}\n\nPerson: '
        prompt = f'{prompt}{message}\n{self.name}: '  # logical error, Person: I talk to the bartender.
        response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0.4,
                                            max_tokens=100, stop='Person:')
        response = response['choices'][0]['text']
        self.former_speak_inputs.append(message)
        self.former_speak_outputs.append(response)
        return response

    def fight(self, message):
        return f'You wrote: "{message}"\nBut fighting {self.name} doesn\'t work.\n(Violence is never a solution)'

    def __call__(self, message):
        prompt = f'{self.description}\n\nDescribe {self.name} and answer questions about {self.name}.\n\nQuestion: '
        for i in range(min(len(self.former_describe_inputs), 3)):
            prompt += f'{self.former_describe_inputs[i]}\nAnswer: {self.former_describe_outputs[i]}\n\nQuestion: '
        prompt += f'{message}\n\nAnswer: '
        response = openai.Completion.create(model="text-davinci-002", prompt=prompt, temperature=0.4,
                                            max_tokens=100, stop='Question:')
        response = response['choices'][0]['text']
        self.former_describe_inputs.append(message)
        self.former_describe_outputs.append(response)
        return response


class SECRET:
    def __init__(self, name, activation_flag, prompt=None):  # "where to find" system?
        self.name = name
        self.prompt = prompt
        self.activation_flag = activation_flag
        self.found = False


class FLAG:  # like secrets without text. For example "won" is now a flag not a secret anymore.
    def __init__(self, name, value=True, conditions=None):  # "and" and "or" and "not" combinations work;
        self.name = name
        self.value = value  # boolean
        self.conditions = conditions
        # conditions consist of flags (true), secrets (found), npcs (in stage), locations (in stage)
        # conditions should be rather complex

    def check(self, stage):
        def check_item(item):  # this function checks for a "not" connection
            if isinstance(item, dict):
                if not len(list(item.keys())) == 1:
                    raise ValueError
                if check_item2(list(item.keys())[0]) == list(item.values())[0]:
                    return True
                else:
                    return False
            else:
                return check_item2(item)

        def check_item2(item):
            if isinstance(item, FLAG):
                return item.value
            elif isinstance(item, SECRET):
                return item.found
            elif isinstance(item, NPC):
                if item in stage['npcs']:
                    return True
                else:
                    return False
            elif isinstance(item, LOCATION):
                if item == stage['location']:
                    return True
                else:
                    return False
            else:
                raise ValueError

        def check_list():
            # This function checks any amount of combinations of "and" and "or" connections
            new_list = [[]]  # list of lists of boolean values
            pos = [0]  # position (of item x) within the conditions
            while True:
                x = self.conditions
                maxs = []  # same length as pos
                for i in pos:
                    maxs.append(len(x))
                    x = x[i]
                if isinstance(x, list):
                    pos.append(0)
                    new_list.append([])
                else:
                    val = check_item(x)  # what to do now with this?
                    new_list[-1].append(val)
                    pos[-1] += 1
                    while maxs[-1] == pos[-1]:
                        vals = new_list.pop()
                        if pos[0] == maxs[0]:  # check if it's done
                            if False in vals:
                                return False
                            else:
                                return True
                        if (len(new_list) % 2) == 0:  # "and" connection
                            if False in vals:
                                new_list[-1].append(False)
                            else:
                                new_list[-1].append(True)
                        else:  # "or" connection
                            if True not in vals:
                                new_list[-1].append(False)
                            else:
                                new_list[-1].append(True)
                        pos.pop()
                        maxs.pop()
                        pos[-1] += 1

        if self.conditions is None or self.conditions == []:
            self.value = self.value
        elif isinstance(self.conditions, list):
            self.value = check_list()
        elif isinstance(self.conditions, bool):
            self.value = self.conditions
        else:
            self.value = check_item(self.conditions)


class TRIGGER:
    def __init__(self, name, activation_flag, call_flag, func):
        self.name = name
        self.activation_flag = activation_flag
        self.call_flag = call_flag  # two flags in case one is "static" and changed by the trigger or so.
        self.func = func  # a string of the function name

    def __call__(self, game):
        func = get_func(game.adventure.name, self.func)
        # count = game.trigger_count
        return func(game)


if __name__ == "__main__":
    pass
