# Class Caial is called by the flask server and sends requests to the openai API

import openai
from os import getenv
from datetime import datetime
from json import loads, decoder
from dotenv import load_dotenv
from pathlib import Path


class Caial:

    root_path = Path(__file__).parents[2]

    # method to create a response with a model, max response tokens, temperature, and message array
    def create_response(
        self,
        mod: str,
        tok: int,
        tmp: float,
        msg: list[dict[str, str]]
    ):
        response = openai.ChatCompletion.create(
            model=mod,
            max_tokens=tok,
            temperature=tmp,
            n=1,
            messages=msg
        )
        return response

    # add metadata to response JSON
    def add_metadata(
        self,
        conversation: dict,
        model: str,
        response,
        code: str,
        programming_language: str,
        success: bool
    ) -> dict:
        conversation["date"] = datetime.now().isoformat()
        conversation["model"] = model
        conversation["total_tokens"] = response.usage.total_tokens
        conversation["completion_tokens"] = response.usage.completion_tokens
        conversation["code"] = code
        conversation["programmingLanguage"] = programming_language
        conversation["success"] = success
        return conversation

    def create_message(self, role: str, message: str) -> dict[str, str]:
        return {"role": role, "content": message}

    # initail prompts + few-shot example
    def create_prompt(self, programming_language: str, code: str) -> list[dict[str, str]]:

        # Python prompt
        if programming_language == "python":
            # load few-shot example
            with open(f"{self.root_path}/assets/PythonExamples/guessinggame.py") as c:
                example = c.read()
            # load few-shot answer
            with open(f"{self.root_path}/assets/JSON/guessinggameCaial.json") as g:
                example_res = g.read()

            messages = [
                self.create_message(
                    "system", f"Your task is to review code in the {programming_language} programming language and your purpose is to give helpful messages regarding coding mistakes or bad habits. You always answer in the JSON format, which contains the fields 'lineFrom', 'lineTo' and 'message'. The message field contains the criticism of the code between the fields lineFrom and lineTo. The message can not include inconsistent Indentations or missing docstrings."),
                self.create_message(
                    "user", f"Here is some Python code:\n{example}"),
                self.create_message("assistant", example_res),
                self.create_message(
                    "user", f"Great response! Here is some more {programming_language} code:\n{code}")
            ]
            return messages

        # ABAP prompt
        elif programming_language == "abap":
            # load example layout
            with open(f"{self.root_path}/assets/JSON/Layout.json", "r") as l:
                layout = l.read()

            messages = [
                self.create_message(
                    "system", f"Your task is to review code in the {programming_language} programming language and your purpose is to give helpful messages regarding coding mistakes or bad habits. You always answer in the JSON format, which contains an array inside the field 'caial'. All objects inside 'caial' contain the fields 'lineFrom', 'lineTo' and 'message'. The message field contains the criticism of the code between the fields lineFrom and lineTo. The message can not include inconsistent Indentations or missing docstrings. Your layout should look like this: {layout}"),
                self.create_message(
                    "user", f"Here is some {programming_language} code:\n{code}")
            ]
            return messages

        # unsupported programming language
        else:
            raise Exception("Unsupported programming language!")

    def get_conversation(self):
        return self.conversation

    # Constuct containing API request
    def __init__(self, programming_language: str, code: str, model: str, max_tokens: int):

        self.success = False

        # Select the GPT model ("gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k" ...)
        # model = "gpt-3.5-turbo-16k"

        # Select the maximum response tokens
        # max_tokens = 2000

        # load the openai api key from the file "API-Key.env"
        load_dotenv(f"{self.root_path}/API-Key.env")
        key = getenv("OPENAI_KEY")
        openai.api_key = key

        # initail prompts + few-shot example
        try:
            messages = self.create_prompt(
                programming_language=programming_language, code=code)
        except Exception as exception:
            print(exception)
            quit()

        # generate response
        print("Generating response ...")
        response1 = self.create_response(
            mod=model, tok=max_tokens, tmp=0.1, msg=messages)

        for choices1 in response1["choices"]:
            try:
                print(response1)
                print("\n========================================\n")
                self.conversation = loads(choices1.message.content)
                self.conversation = self.add_metadata(
                    conversation=self.conversation,
                    model=model,
                    response=response1,
                    code=code,
                    programming_language=programming_language,
                    success=True
                )
                self.success = True
                print("Successful after one try.")

            except decoder.JSONDecodeError:
                print("Response is not in JSON!\nRetrying ...")
                messages.append(self.create_message(
                    "assistant", choices1.message.content))
                messages.append(self.create_message(
                    "user", "Your response was not a valid JSON. Please try again without any strings attached to your response."))

                # creating 2nd response
                response2 = self.create_response(
                    mod=model, tok=max_tokens + response1.usage.completion_tokens, tmp=0.1, msg=messages)

                for choices2 in response2["choices"]:
                    try:
                        print(response2)
                        print("\n========================================\n")
                        self.conversation = loads(choices2.message.content)
                        self.conversation = self.add_metadata(
                            conversation=self.conversation,
                            model=model,
                            response=response2,
                            code=code,
                            programming_language=programming_language,
                            success=True
                        )
                        self.success = True
                        print("Successful after two tries.")
                    except decoder.JSONDecodeError:
                        print("Response is not in JSON again!\nStopped!")
