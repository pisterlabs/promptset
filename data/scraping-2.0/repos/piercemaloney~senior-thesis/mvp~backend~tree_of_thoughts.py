from openai import call_openai_gpt3
from typing import List
from prompts import generate_prompt, vote_prompt

def vote(current_proof: str, tactics: str):
    """tactics_list is a comma separated string of tactics"""
    tactics = tactics.split(", ")
    proposed_steps = "\n".join([f"{i+1}) {tactic}" for i, tactic in enumerate(tactics)])
    result = call_openai_gpt3(vote_prompt.format(input=current_proof, proposed_steps=proposed_steps))
    return proposed_steps[int(result) - 1]
    
def generate(current_proof: str):
    return call_openai_gpt3(generate_prompt.format(input=current_proof))
