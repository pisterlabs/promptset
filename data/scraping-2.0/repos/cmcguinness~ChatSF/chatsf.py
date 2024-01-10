"""

    ChatSF:

    A Simple Chatbot that connects OpenAI and Salesforce to answer questions
    about Salesforce data.  This file drives


"""
#   Standard imports
import sys
import textwrap
import time

#   My modules
from gpt import OpenAI
from plog import Plog
from salesforce import SalesforceFunctions


#
#   I put this into a class just because it was easier to organize the data
#   The expectation is that there is just one chatbot instance at a time
#
class ChatBot:

    def __init__(self):

        #   When this returns, we have all the user information in the instance
        self.sf = SalesforceFunctions()

        #   Which we then use to augment the prompt so GPT knows who's talking to it
        self.system_prompt = f"""
            You answer questions about CRM data which you can do using SQL queries.
            You give short and to the point answers to the questions.  

            The user talking to you has the following details:
            
            {{ "name": "{self.sf.user_name}", "email": "{self.sf.user_email}", "company": "{self.sf.user_companyname}",
            "phone": "{self.sf.user_phone}", "title": "{self.sf.user_title}", "UserId": "{self.sf.user_id}" }}

            You can use this information to fill in your answers if needed.
        """
        #   Because of the way indents get put into multi-line string constants in python,
        #   and because I want to make prompts as short as possible, I do this to take them
        #   apart, trim off leading/trailing spaces from each line, and reassemble the prompt
        lines = self.system_prompt.split('\n')
        self.system_prompt = '\n'.join([l.strip() for l in lines])

        Plog.debug('System Prompt:\n' + self.system_prompt)

        # Initialize our GPT class
        self.gpt = OpenAI(self.system_prompt, self.sf, timeout=30)

        print('Welcome to ChatSF.  You may ask questions about accounts, opportunities, and contacts')

    #   In order display long blocks of text, we want to make sure
    #   that we've word-wrapped each line so it doesn't scroll horizontally
    @staticmethod
    def rewrap_text(text: str, maxlen=80) -> str:
        textlines = text.splitlines()
        newlines = []
        for line in textlines:
            if len(line) <= maxlen:
                newlines.append(line)
            else:
                wrapped = textwrap.wrap(line, maxlen)
                for w in wrapped:
                    newlines.append(w)

        return '\n'.join(newlines)

    @staticmethod
    # If the output from GPT is a single line, just print it, otherwise
    # print each line with a leading indent
    def bot_print(text):
        if '\n' not in text:
            print('\nGPT:  ' + text)
        else:
            lines = text.splitlines()
            print('\nGPT:')
            for line in lines:
                print('      ' + line)
        print('', flush=True)

    #
    #   autobot:
    #
    #   Given a list of user inputs, run through them as if the user had
    #   entered them
    #
    def autobot(self, user_inputs: [], show_pony_mode: bool):
        for test in user_inputs:
            if show_pony_mode:
                # This makes it look like a human is typing in the text
                print('User: ', end='', flush=True)
                time.sleep(2)
                for c in test:
                    print(c, end='', flush=True)
                    time.sleep(.05)
                time.sleep(2)
                print('', flush=True)
            else:
                print('User: ' + test, flush=True)

            g = self.rewrap_text(self.gpt.ask_gpt(test))

            self.bot_print(g)

    # This is our normal, interactive ChatBot
    def chatbot(self):
        while True:
            u = input('\nUser: ')
            g = self.rewrap_text(self.gpt.ask_gpt(u))

            self.bot_print(g)


if __name__ == '__main__':
    test_suite = [
        "What are my open opportunities?",
        "What are my opportunities at Edge?",
        "Tell me about the generator opportunity?",
        "Tell me a bit about Sean",
        "What are the three best things I could do to make sure Sean is focusing on our deal?",
        "Could you write the email for me?",
        "What other opportunities do I have open?",
        "Tell me more about the genepoint deal",
        "Who are the contacts on the deal?",
        "Compose an email to her asking if there's anything else she'd like us to do to help close the deal"
    ]
    Plog.set_level('Info')
    bot = ChatBot()

    # Comment this out / in to run the test suite
    bot.autobot(test_suite, True)

    bot.chatbot()
