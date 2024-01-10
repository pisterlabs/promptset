import os, sys
import openai_service, response_handler, languages
from termcolor import colored

class Main():

    VERSION="0.1.0"
    SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))
    KEY_FILE="../.env"
    KEY_NAME="OPENAI_API_KEY"

    def __init__(self):
        self.message("Welcome to ChatGP-Tea!")
        self.message("This is a chatbot that uses the OpenAI API to generate responses to your questions.")

        print(colored(f"\nVersion: {self.VERSION}", "magenta"))

        print("\nCommands:")
        print("- ask    : Ask a question.")
        print("- lang   : Ask a question in a specific language.")
        print("- convo  : Start a conversation with the bot.")

        API_KEY = os.environ.get(self.KEY_NAME)
        if API_KEY == None:
            if os.path.exists(f"{self.SCRIPT_PATH}/{self.KEY_FILE}"):
                with open(f"{self.SCRIPT_PATH}/{self.KEY_FILE}", "r") as f:
                    API_KEY = f.read().replace(f"{self.KEY_NAME}=", "")
            else:
                self.error("No API key found.")
                sys.exit()
        self.openaiService = openai_service.OpenAIService(API_KEY)

    def process_args(self):
        args = sys.argv[1:]
        n = len(args)
        if n > 0:
            arg0 = args[0]
            if arg0 == "ask":
                self.get_response()
            elif arg0 == "lang":
                if n > 1:
                    self.get_lang_response(args[1])
                else:
                    self.error("No language specified.")
            elif arg0 == "convo":
                self.start_convo()                
            else:
                self.error(f"Invalid argument: {args[0]}")
        else:
            self.info("Type a command to get started!\n")

    def start_convo(self):
        self.info("User:")
        question = input()

        while question != "exit":
            response = self.openaiService.get_response(question)
            self.info("ChatGPT:")
            print(f"{response}")
            self.info("User:")
            question = input()

        self.message("\nWould you like to save this conversation? (y/n)")
        save = input()
        if save == "y":
            self.message("\nWhat would you like to call this conversation?")
            fname = input()
            self.openaiService.save_conversation(fname)
        
        self.message("\nGoodbye!")

    def get_response(self):
        response = main.ask_question()
        self.info("ChatGPT:")
        print(f"\n{response}")

    def get_lang_response(self, lang):
        lang = languages.Languages().get_language(lang)
        if lang is not None:
            response = main.ask_lang_question(parseLanguage=lang)
            main.format_lang_response(response, lang=lang)
        else:
            self.error(f"Invalid language: {lang}")

    def ask_question(self):
        self.info("User:")
        question = input()

        return self.openaiService.get_response(question)

    def ask_lang_question(self, parseLanguage=languages.Languages().get_language("python"), executeLanguage="bash"):
        self.info(f"What would you like your {executeLanguage} command to do?")
        question = input()

        return self.openaiService.get_lang_response(question, parseLanguage)
    
    def format_lang_response(self, response, lang=languages.Languages().get_language("python")):
        responseHandler = response_handler.ResponseHandler(response)
        code_blocks = responseHandler.findCodeBlocks(lang=lang)

        if len(code_blocks) == 0:
            self.error("No code found in response.")
            self.info("ChatGPT:")
            print(f"\n{response}")
            sys.exit()
        else:
            responseHandler.save_code_blocks(code_blocks, lang)
            self.info("ChatGPT:")
            print(f"\n{response}")

    def error(self, msg):
        print(colored(f"\nERROR: {msg}", "red"))

    def info(self, msg):
        print(colored(f"\n{msg}", "green"))

    def message(self, msg):
        print(colored(f"{msg}", "blue"))


if __name__ == "__main__":
    main = Main()
    main.process_args()