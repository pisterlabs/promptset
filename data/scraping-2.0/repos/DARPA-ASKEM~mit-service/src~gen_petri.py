from gpt_interaction import *
from openai import OpenAIError
import re

def get_places(text, gpt_key):
    try:
        prompt = get_petri_places_prompt(text)
        match = get_gpt_match(prompt, gpt_key)
        #print(match)
        places = match.split(":")[-1].split(",")
        return places, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

def get_transitions(text, gpt_key):
    try:
        prompt = get_petri_transitions_prompt(text)
        match = get_gpt_match(prompt, gpt_key)
        #print(match)
        places = match.split(":")[-1].split(",")
        return places, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

def get_arcs(text, gpt_key):
    try:
        prompt = get_petri_arcs_prompt(text)
        match = get_gpt_match(prompt, gpt_key, "text-davinci-003")
        #print(match)
        lines = match.splitlines()
        transitions = []
        for line in lines:
            words = [w.rstrip() for w in line.split("->")]
            if (len(words) == 0):
                continue
            transitions.append(words)
        return transitions, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

def match_place_to_text(text, place, gpt_key):
    try:
        prompt = get_petri_match_place_prompt(text, place)
        match = get_gpt_match(prompt, gpt_key)
        #print(match)
        #places = match.split(":")[-1].split(",")
        return match, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

def init_param_from_text(text, param, gpt_key):
    try:
        prompt = get_petri_init_param_prompt(text, param)
        match = get_gpt_match(prompt, gpt_key)
        return match.replace(")", ") ").split(" "), True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False


def match_place_and_text_to_columns(place, text, columns, gpt_key):
    try:
        prompt = get_petri_match_dataset_prompt(place, text, columns)
        match = get_gpt_match(prompt, gpt_key)
        #print(match)
        #places = match.split(":")[-1].split(",")
        return match, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

if __name__ == "__main__":
    gpt_key = ""
    with open("../resources/jan_hackathon_scenario_1/SEIRD/seird.py", "r") as f:
        code = f.read()
    with open("../resources/jan_hackathon_scenario_1/SEIRD/section2.txt", "r") as f:
        text = f.read()
    with open("../resources/jan_hackathon_scenario_1/SEIRD/sections34.txt", "r") as f:
        text2 = f.read()
    with open("../resources/dataset/headers.txt", "r") as f:
        columns = f.read()[:3000]


    places, s = get_places(code, gpt_key)
    parameters, s = get_parameters(code, gpt_key)
    transitions, s = get_transitions(code, gpt_key)

    print(f"places:\t\t{places}\n------\n")
    print(f"parameters:\t\t{parameters}\n------\n")
    print(f"transitions:\t\t{transitions}\n------\n")

    for place in places:
        desc, s = match_place_to_text(text, place, gpt_key)
        print(f"description of {place}: {desc}\n------\n")
        cols, s = match_place_and_text_to_columns(place, text, columns, gpt_key)
        print(f"Columns for {place}: {cols}\n------\n")

    for param in parameters:
        val, s = init_param_from_text(text2, param, gpt_key)
        print(f"Initial value of {param}: {val}\n------\n")