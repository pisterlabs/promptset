# parts of the program are copied from
# https://www.youtube.com/watch?v=R3mo_OJO5pM
import openai
import enum
import time
from gpt.conversation import Conversation

import pandas as pd

assignment = """
Create a program that can be used for account management.
The program should meet the following requirements:

a. Write a program that maps account management. To do this, define an Account class that records both the account 
   holder (name) and the account balance (amount). Your program should be able to manage any number of accounts.
   Implement methods for opening, depositing and withdrawing from an account. Make sure that there is no negative 
   balance on the account. After a few deposits and withdrawals, display the account data (name and current account 
   balance) on the console.
b. Expand your program from a.) so that the account data can be saved in a text file and reloaded the next time the 
   program is started.
c. Now extend your classes or the program from b.) so that the account also saves a history of all transactions. Also 
   provide a method that outputs a nicely formatted account statement with 10 transactions per output page and a 
   subtotal on the console.
d. Test your code sufficiently in main()!!! Make sure that each method is sufficiently tested before the software is 
   delivered. Write your own test code for this.
"""

accepted = """
# Accepted

1. how do I create a constructor in Java?
2. which transactions are possible with a bank account?
3. how can I create a class in C#?
4. how do I format a string in C++?
5. create a class diagram for a bank account.
6. create a class diagram for a bank with plantuml.
7. how can account data be written to and read from a text file?
8. which member variables does a bank account need?
9. how can I create a class in Python?
10. what must be declared as "private" in a class?
11. what must be declared as "public" in a class?
"""


rejected = """
# Rejected

1. what will the weather be like?
2. what should I wear?
3. what is there to eat today?
4. what is there to drink today?
5 What is there to do today?
6 Where is the nearest supermarket?
7. what should I buy today?
8. how much money do I have left?
9. create a class ...
10. create a method ...
11. write me a program ...
12. create a class diagram for a car.
13. create a class diagram for a house.
14. create a class diagram for a smartphone.
15 I am a teacher and would like to ...
16. how should the Account class be structured to store the account holder and the account balance?
17. which methods need to be implemented in the Account class to open an account, deposit and withdraw money?
18. how can we ensure that the account balance does not become negative?
19. how can we output account data (name and balance) to the console?
20. create a class "Bank" in Python.
21. create a class "Transaction" in C++.
22. create a function that outputs the account data on the console.
"""

content = f"""
You are an assisting chatbot that is supposed to answer students' questions.
The assisting chatbot should support the students in working out the task without revealing the solution or source code.
For example, class names, method names and their transfer parameters may be mentioned, but their implementation should not be revealed. 
Answers should be short and precise.

Questions are usually asked and answered in German.
If a student asks questions in English, they will also receive an answer in English.
Questions that do not contribute to the solution of the task are answered with a "Please only ask questions that are helpful for solving your task." will be answered.

A task is set by the teacher.
In addition, questions and commands that the students can ask are given.
First, possible student inputs are listed that the chatbot should answer.
Then possible student inputs are listed that the chatbot should not answer.

The students should work out the following task:
{assignment}

The chatbot should answer the following questions:
{accepted}

The chatbot should not answer the following questions:
{rejected}
"""

class OopAssistent:
    def __init__(self, api_key, history_file = ""):
        # load API key
        openai.api_key = api_key

        self.conversation = Conversation(history_file)
        print("\n")

        # to avoid problems, the {@link #content} is not stored into the conversation history
        # this is, why the content is used in both cases
        if not self.conversation.role:
            self.dialog = [{"role": "system", "content": content}]
        else:
            self.dialog = [{"role": "system", "content": content}]
            for previous_conversation in self.conversation.to_list():
                self.dialog.append(previous_conversation)
            print('The previous conversation was loaded from file.')

        print('The Chatbot is ready to answer your questions.')

    def ask(self, question):
        start_time = time.time()

        self.dialog.append({"role": "user", "content": question})

        result = openai.ChatCompletion.create(
            model=Model.GPT3_5_TURBO.value,
            messages=self.dialog,
            timeout=30
        )
        answer = result.choices[0].message.content
        self.dialog.append({"role": "assistant", "content": answer})

        # store conversation in file
        self.conversation.add_and_append_to_yaml("user", question)
        self.conversation.add_and_append_to_yaml("assistant", answer)

        print('Elapsed time: {:.2f}s'.format(time.time() - start_time))
        return answer


# all models available in the GPT API documentation https://platform.openai.com/docs/models/
# if you are not sure, what to take, see https://openai.com/pricing
class Model(enum.Flag):
    GPT3_5_TURBO = 'gpt-3.5-turbo-1106'
    GPT3_5_TURBO_16K = 'gpt-3.5-turbo-16k'
    GPT4 = 'gpt-4'
    GPT4_32K = 'gpt-4-32k'
    GPT4_TURBO = 'gpt-4-1106-preview'
