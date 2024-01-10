import openai
import time
from tqdm import tqdm
import argparse
import configparser
import os

from agent import Agent

parser = argparse.ArgumentParser(
    description='conduct a debate to get conclusions of a subject.')
# inputs
parser.add_argument('--subject', type=str, default='Should we regulate the use of large language models in academic research?',
                    help='Please provide the subject for a debate.')
# functions
# outputs
parser.add_argument('--output', type=str, default='eg_dialogue',
                    help='Export to the file.')

args = parser.parse_args()

# set config of OpenAI
# reference. https://www.andreagrandi.it/2023/05/05/use-openai-api-to-make-two-gpt-bots-talk/
api_keyA = input('Your OpenAI API key: ')  # OpenAI API key 1 TODO. config
api_keyB = input('Yout OpenAI API key: ')  # OpenAI API key 2 TODO. config

# set config of agents
subject = args.subject
output = args.output
# model: https://stackoverflow.com/questions/75396481/openai-gpt-3-api-error-this-models-maximum-context-length-is-4097-tokens
agent_configA = {'name': 'Agent-A', "argument_strength": '0.9',
                 'llm_config': {'model': 'gpt-3.5-turbo-16k'}}  # TODO. flexible w\ UI
agent_configB = {'name': 'Agent-B', "argument_strength": '0.9',
                 'llm_config': {'model': 'gpt-3.5-turbo-16k'}}  # TODO. flexible w\ UI

# build agents
agentA = Agent(name=agent_configA['name'], api_key=api_keyA,
               system_message=None, llm_config=agent_configA['llm_config'])
agentB = Agent(name=agent_configB['name'], api_key=api_keyB,
               system_message=None, llm_config=agent_configB['llm_config'])
# print the current round of the debate


def print_round(t):
    t = t + 1
    print(t)
    return t


# Continuously loop through multiple rounds of conversation
debate_round = 0
while True:
    print('='*32)
    # Talk to which agent or exit.
    command_message = input('System command (A/B/save/round/exit): ')
    if command_message == "exit":  # Exit the loop
        break
    elif command_message == 'round':
        debate_round = print_round(debate_round)
        continue
    elif command_message == 'debug':
        breakpoint()
        continue
    elif command_message not in ('A', 'B', 'save', 'round', 'exit'):
        print('Unknown command. Please try again.')
        continue

    if command_message == 'save':
        responseA = agentA.get_last_messages()
        responseB = agentB.get_last_messages()
        print('Successfully save agent A\'s and B\'s latest reply.')
        continue

    user_message = input("A moderator: ")  # Get user input
    with open(f'{output}.log', 'a') as f:
        f.write(f'A moderator: {user_message}\n\n')

    if '[cp]' in user_message:
        cp_agent = input('Copy and paste results from (A/B/AB/BA): ')
        if cp_agent == 'A':
            cp_message = f'\n(Messages from Agent-A): {responseA}\n'
        elif cp_agent == 'B':
            cp_message = f'\n(Messages from Agent-B): {responseB}\n'
        elif cp_agent == 'AB':
            cp_message = f'\n(Messages from Agent-A): {responseA}\n(Messages from Agent-B): {responseB}\n'
        elif cp_agent == 'BA':
            cp_message = f'\n(Messages from Agent-B): {responseB}\n(Messages from Agent-A): {responseA}\n'

        user_message = user_message.replace('[cp]', cp_message)
    elif '[exit]' in user_message:
        print(f'Ignore this message to Agent-{command_message}.')
        continue

    print()

    if command_message == 'A':
        # Add the user's message to the conversation
        reply = agentA.receive(user_message, sender='human', request_reply=1)
        reply = f'Agent-A: {reply}\n'

        print()
    elif command_message == 'B':
        # Add the user's message to the conversation
        reply = agentB.receive(user_message, sender='human', request_reply=1)
        reply = f'Agent-B: {reply}\n'
        print()

    with open(f'{output}.log', 'a') as f:
        f.write(f'{reply}\n')
