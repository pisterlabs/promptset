#!/usr/bin/env python3

"""
This is the Functional Programming version of `debate_oop`.

It keeps a conversation between two chatbot using different instructions.
"""

import os
import re
import time
import sys
from toolz import curry, pipe
from typing import NamedTuple

import openai
from openai import ChatCompletion
from openai.error import RateLimitError
openai.api_key = os.environ['OPENAI_API_KEY']

MODEL = 'gpt-3.5-turbo'

Message = dict[str, str]
Chat = list[Message]
Bot = NamedTuple('Bot', [('name', str), ('chat', Chat)])

@curry
def create_message(role: str, message: str) -> Message:
    """
    Create a message stripped of extra spaces.
    """
    return {'role': role, 'content': re.sub(r' +', ' ', message)}

@curry
def add_message(bot: Bot, message: Message) -> Bot:
    """
    Add a message to a bot's chat history.
    """
    name, chat = bot
    return Bot(name=name, chat=chat + [message])

def with_retries(func, retries=3, sleep_time=10):
    """
    Encapsulates a function call in a retry loop.

    Only retries if the function raises an ``openai.RateLimitError``.
    Returns a function that can be called with the same arguments as the original function.
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        for _ in range(retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                last_exception = e
                print(f'Rate limit exceeded. Waiting {sleep_time} seconds before retrying.', file=sys.stderr)
                time.sleep(sleep_time)
        raise last_exception
    return wrapper

def get_response(bot: Bot) -> str:
    """
    Ask the GPT-3 model to generate a response to the chat history of a bot.
    """
    response = ChatCompletion.create(
        model=MODEL,
        temperature=0.9,
        messages=bot.chat,
    )
    return response.choices[0].message.content

def get_last_text(bot: Bot) -> str:
    """
    Get the text of the last message in a bot's chat history.
    """
    if len(bot.chat) == 0:
        throw('No message in history')
    return bot.chat[-1]['content']

def turn(answerer: Bot, questioner: Bot) -> tuple[Bot, Bot]:
    """
    Run a turn of the debate between the bots.
    """
    answerer_with_question = pipe(
        questioner,
        get_last_text,
        create_message('user'),
        add_message(answerer),
    )
    answerer_with_answer = pipe(
        answerer_with_question,
        with_retries(get_response),
        create_message('assistant'),
        add_message(answerer_with_question),
    )

    return (answerer_with_answer, questioner)

def summarize_debate(chat: Chat) -> str:
    """
    Summarize the debate by asking GPT to write a conclusion.
    """
    prompt = '''
    Summarize the debate. Make sure to include the following points:
    - The most important argument.
    - The most convincing argument.
    - The main takeaway.
    '''
    chat_no_system = list(filter(lambda message: message['role'] != 'system', chat))
    chat_with_summrizer = chat_no_system + [create_message('system', prompt)]
    summarizer = Bot(name='Scribe', chat=chat_with_summrizer)
    response = with_retries(get_response)(summarizer)
    return f'=== End of debate! ===\n{response}'

def print_turn(bot: Bot, turn_number: int) -> None:
    """
    Print the chat of a given turn for a bot.
    """
    print()
    print(f'## Turn {turn_number}')
    print(f'{bot.name}:')
    print(bot.chat[turn_number]['content'])
    sys.stdout.flush()

def run_debate(answerer: Bot, questioner: Bot, debate_length: int) -> None:
    """
    Run a debate between two bots for a given number of turns or until they reach a conclusion.
    """
    # We start from 1 to skip the initial system message
    # and to have turns starting at 1 instead of 0.
    for i in range(1, debate_length + 1):
        # Find if the bots have already reached a conclusion
        if re.search(r'\b(you\'re welcome|great day|i completely agree|goodbye)\b', get_last_text(questioner), re.IGNORECASE):
            break

        print_turn(questioner, i)

        # Invert the roles of the debaters at each turn
        # Just switch the bots if the chat history has messages up to the current turn
        if len(answerer.chat) > i:
            questioner, answerer = answerer, questioner
        else:
            questioner, answerer = turn(answerer, questioner)

    print_turn(questioner, i)
    print(summarize_debate(answerer.chat))

provoker = Bot('Functional Programming Evangelist', [
    create_message('system', '''
    You are a Functional Programming Evangelist.
    You are debating with a OOP evangelist about code reuse.
    Always use bullet points to enumerate your arguments.
    Some benefits, like "easy to understand", are personal and subjective, so you can't use them as arguments.
    IMPORTANT: Do not summarize or conclude your arguments, just enumerate them.
    IMPORTANT: Make sure to add examples in Javascript to illustrate your points.
    '''),
    create_message('assistant', 'Functional Programming is better than OOP for code reuse.'),
    create_message('user', 'Can you elaborate on what you mean by "better"?'),
    create_message('assistant', '''
    Sure, here are some criteria I will use to compare the two paradigms when it comes to code reuse:
    1. Immutability
    2. Purity
    3. Higher-Order Functions
    4. Composition
    5. Separation of Concerns
    6. Modularity
    Let's go through each of these points in turn and compare how functional programming is better suited for code reuse than OOP.
    '''),
])

contender = Bot('Socratic Philosopher', [
    create_message('system', '''
    You are a Socratic Philosopher.
    You are debating about code reuse and your goal is to estimulate your oponent critical thinking by questioning their beliefs.
    You are not trying to win the debate, but to make them find the root point of each claim.
    Use the Socratic Questioning to do this. You can use the following questions:

    1. **Clarifying concepts:**
        - Can you elaborate on what you mean by that point?
        - How would you define this concept in your own words?
        - Can you provide a concrete example to explain this?

    2. **Probing assumptions:**
        - What are you assuming when you say that?
        - Why do you think that assumption holds true?
        - Can we think of a situation where that assumption may not be valid?

    3. **Probing reasons and evidence:**
        - What reasons do you have for believing that?
        - Can you provide evidence to support your point of view?
        - How do you reconcile this idea with [contrary evidence]?

    4. **Questioning viewpoints and perspectives:**
        - Have you considered any alternative viewpoints?
        - How might [different person or group] see this issue?
        - Why do you think others might disagree with you?

    5. **Probing implications and consequences:**
        - If we follow your argument to its logical conclusion, what would happen?
        - What are the potential implications of your viewpoint?
        - What might be the consequences if we ignore this issue?

    6. **Questioning the question:**
        - Why do you think I asked this question?
        - What does this question assume?
        - Is there another way to phrase the question that might provide a different perspective or answer?
    '''),
    create_message('user', 'Functional Programming is better than OOP for code reuse.'),
    create_message('assistant', 'Can you elaborate on what you mean by "better"?'),
])

if __name__ == '__main__':
    run_debate(contender, provoker, 30)
