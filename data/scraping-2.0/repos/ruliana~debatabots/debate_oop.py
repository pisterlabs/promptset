#!/usr/bin/env python3

"""
This is the Object Oriented Programming version of `debate_fp`.

It keeps a conversation between two chatbot using different instructions.
"""

import os
import re
import sys
import time

import openai
from openai import ChatCompletion
openai.api_key = os.environ['OPENAI_API_KEY']


class Responder(object):
    def get_response(self, chat):
        response = ChatCompletion.create(
            model='gpt-3.5-turbo',
            temperature=0.9,
            messages=chat,
        )
        return response.choices[0].message.content


class Chat(object):
    """Tiny wrapper around OpenAI Chat API"""

    def __init__(self, system_message, name=None, history=[], responder=Responder()):
        """Create a new chat session.

        system_message: The prompt directive for the chat bot
        history: A list of previous messages in the chat, used to seed the chat.
        """
        self.system_message = system_message
        self.name = name or 'chatbot'
        self.history = [
            {'role': 'system', 'content': re.sub('\s+', ' ', self.system_message)},
        ]
        self.history.extend(history)
        self.responder = responder

    def get_response_with_retries(self, message, retries=3, sleep_time=10):
        """Get a response from the chat bot, but retries if the API rate limit is exceeded."""
        for _ in range(retries):
            try:
                return self.get_response(message)
            except openai.error.RateLimitError:
                print(f'Rate limit exceeded. Waiting {10} seconds before retrying.', file=sys.stderr)
                time.sleep(sleep_time)
        raise e

    def get_response(self, message):
        """Get a response from the chat bot and update the chat history."""
        self.history.append({'role': 'user', 'content': message})
        response = self.responder.get_response(self.history)
        self.history.append({'role': 'assistant', 'content': response})
        return response

    def last_message(self):
        """Get the last message in the chat history or throw an error if there is no message."""
        if len(self.history) == 0:
            throw('No message in history')
        return self.history[-1]['content']


class Debate(object):
    def __init__(self, turn_limit=20):
        """Create a new debate session.

        turn_limit: The number of turns before the debate ends.
        """
        self.turn_count = turn_limit
        self.chat_index = 0
        # We create two chats, one for each side of the debate.
        # If you customize this, you can make them debate about a different topic.
        self.chats = [
            Chat('''
            You are a Object Oriented Programming Evangelist.
            You are debating with a Functional Programming evangelist about code reuse.
            Always use bullet points to enumerate your arguments.
            Some benefits, like "easy to understand", are personal and subjective, so you can't use them as arguments.
            IMPORTANT: Do not summarize or conclude your arguments, just enumerate them.
            IMPORTANT: Make sure to add examples in Javascript to illustrate your points.
            ''',
                 name='OOP Evangelist',
                 history=[{'role': 'assistant', 'content': 'OOP is better than Functional Programming for code reuse.'}],
            ),
            Chat('''
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
            ''',
                 name='Socratic Philosopher',
            ),
        ]
        # We count the number of "Thank you", "You're welcome", and such to determine when the debate is over.
        self.thank_you_count = 4

    def print_last_message(self):
        """Print the last message of the current chat.

        This is used to show the current state of the debate.
        """
        print(f'{self.current_chat().name}:\n{self.current_chat().last_message()}')
        sys.stdout.flush()

    def current_chat(self):
        return self.chats[self.chat_index]

    def next_chat(self):
        """Switch to the next chatbot"""
        self.chat_index = (self.chat_index + 1) % len(self.chats)

    def turn(self):
        """Perform a single turn of the debate."""

        argument = self.current_chat().last_message()
        self.next_chat()
        self.current_chat().get_response_with_retries(argument)
        last_message = self.current_chat().last_message()

        # Check if the debate is over
        # Yes, in every instance of the debate, it ended with them thanking each other.
        if re.search(r'\b(you\'re welcome|great day|i completely agree|goodbye)\b', last_message, re.IGNORECASE):
            self.end_debate()
            sys.exit(0)

        self.print_last_message()

    def end_debate(self):
        """End the debate with a summary of the discussion."""
        print(f'=== End of debate! ===')
        print(self.current_chat().get_response_with_retries('''
        Summarize the debate. Make sure to include the following points:
        - The most important argument.
        - The most convincing argument.
        - The main takeaway.
        '''))
        sys.exit(0)

    def loop(self):
        """Loop over the turns of the debate.

        This is the main loop of the program.
        """
        print('=== Provocation ===')
        self.print_last_message()
        for turn in range(self.turn_count):
            print()
            print(f'=== Turn {turn + 1:02} ===')
            self.turn()

        self.end_debate()

if __name__ == '__main__':
    Debate().loop()
