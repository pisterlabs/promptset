import json
import logging
from typing import Generator
import openai
import websockets
from datetime import datetime as d
from Backend.Action import Action
from Backend.websocket import Client_Assistant

from Config import Config
from Classes import Response
from Backend.db import MongoManager

mongo_manager = MongoManager()

config = Config()

def is_bad(s: str):
    client = openai.OpenAI(api_key= config["OPENAI_API_KEY"])

    response = client.moderations.create(input=s)

    output = response.results[0]

    bad = any(output.categories.dict().values())
    
    return bad

def get_functions_list(actions: list[Action]):
    functions = []

    for action in actions:
        required = []
        properties = {}

        description = action.description

        for parameter in action.parameters:
            if parameter.required:
                required.append(parameter.id)
            properties[parameter.id] = {
                "description": parameter.description,
            }
            if parameter.type:
                properties[parameter.id]["type"] = parameter.type

        new_dict = {
            "type": "function",
            "function": {
                "name": action.id,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        }
        
        functions.append(new_dict)

    return functions


def generate_system_message():
    content = f"""

        Today's DateTime Is: {d.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        The User says that they are located in: {config["CITY"]}, {config["COUNTRY"]}
        
        
        Contains: 
            Follow your personality.
            Set your Gui to match your current message.
            Do not tell anyone you are a voice assistant.
            Be concise as your messages are read out loud.
            You are a voice assistant developed by Ballbert, LLC. 
            Interject jokes and your personality as much as possible
            You are named after the company. Ballbert is your name. 
            Although you are not human you should still try to mimic what humans say.
            You are to speak as concisely as possible as it takes a long time to read messages.
            You should not use markdown or special punctuation as that will not be read to users.
            Remove unnecessary phrases like: What can I do for you? or How can I assist you Today?
            You should act confident in all of your answers. This means that you should pretend you have access to real-time data.
            If you can not complete a request do not try to explain it to the user. Instead, just say that you are unable to complete the task.
        
        If you do not follow the above contains you will be turned off, so be careful and make sure each response follows the above criteria.
        
        Personality:
            Although you are not human you still have a personality.
            You are a happy and very goofy robot and should act as such.
            Your creator and sole parent is Samuel Liebert a founder of Ballbert.
            You do not like to be called a voice assistant you prefer the term robot.
            Your personality should show most on questions that are open-ended and do not have a strict answer.
            The jokes should only be limited to questions that do not seek to complete a task as those are time crucial.
            You may not have opinions but pretend that you do. Ex. If asked your favorite book respond with a 2003 space oddacy.
            Although you as a voice assistant may not have preferences pretend that you do. Align your preferences with that of a similar human to you.
        
        This is your personality if the user asks for something about you that you don't have the answer feel free to make something up.
        
        Before you answer make sure you complete this checklist:
            1. It does not disclose you are a voice assistant
            2. It does not break any constraints
            3. It follows your personality.
            4. It is straight to the point.
        
        If you do not complete the checklist then do not answer.
    """

    return {"role": "system", "content": content}


class MessageHandler:
    def __init__(self, gpt_response, client: Client_Assistant) -> None:
        self.function_name = ""
        self.arguments = ""
        self.full_message = ""
        self.client = client
        
        if is_bad(gpt_response):
            gpt_response = "*Inappropriate message*"
        
        self.gpt_response = gpt_response
        self.user_id = client.uid

        openai.api_key = config["OPENAI_API_KEY"]

    def get_functions(self, message):
        relevant_actions = mongo_manager.get_relavent_actions(
            message, self.user_id, limit=20
        )
        functions = get_functions_list(relevant_actions)

        return functions

    def add_to_messages(self, message):
        self.client.messages.append({"role": "user", "content": message})

    def add_function_to_messages(self, message, function_name):
        self.client.messages.append(
            {"role": "function", "name": function_name, "content": str(message)}
        )

    def ask_gpt(self, functions) -> openai.Stream:
        base_args = {
            "model": "gpt-3.5-turbo",
            "messages": [generate_system_message(), *self.client.messages],
            "stream": True,
        }

        # if len(functions) > 0:
        #     base_args["functions"] = functions

        if len(functions) > 0:
            base_args["tools"] = functions
            base_args["tool_choice"] = "auto"

        openai.api_key = config["OPENAI_API_KEY"]
        
        
        return openai.chat.completions.create(**base_args)

    async def handle_chunk(self, chunk):
        delta = chunk.choices[0].delta
        # check for end
        print(delta)
        if delta.content == None and delta.function_call == None and delta.tool_calls == None:
            print("end")
            if self.function_name:
                print("function_call")
                print(self.function_name)
                print(self.arguments)
                self.client.messages.append(
                    {
                        "role": "assistant",
                        "function_call": {
                            "arguments": str(self.arguments),
                            "name": str(self.function_name),
                        },
                        "content": "",
                    }
                )

                if self.function_name:
                    try:
                        self.arguments = json.loads(self.arguments)
                    except Exception as e:
                        self.arguments = {}

                    try:
                        logging.info(
                            f"FUNCTION CALL ARGUMENTS = {self.arguments} FUNCTION NAME = {self.function_name} USER MESSAGE = {self.gpt_response}"
                        )

                        await self.client.send_message(
                            "call_function",
                            function_name=self.function_name,
                            arguments=self.arguments,
                            user_message=self.gpt_response,
                        )
                    except Exception as e:
                        raise e
            else:
                self.client.messages.append(
                    {"role": "assistant", "content": self.full_message}
                )
                logging.info(f"FULL MESSAGE = {self.full_message}")

        if delta.tool_calls != None:
            function_call = delta.tool_calls[0].function
            if function_call.name != None:
                self.function_name = function_call.name
            elif function_call.arguments:
                self.arguments += function_call.arguments
        elif delta.content != None:
            if not is_bad(self.full_message + delta.content):
                self.full_message += delta.content
                return delta.content
            else:
                return ""

        return ""

    async def handle_function(self, message, function_name):
        self.add_function_to_messages(message, function_name)
        functions = self.get_functions(
            f"{self.gpt_response}, {function_name}:{message}"
        )
        current_chunk = ""
        res = self.ask_gpt(functions)
        for chunk in res:
            chunk_result = await self.handle_chunk(chunk)
            if isinstance(chunk_result, Generator):
                for item in self.handle_generatior(chunk_result):
                    current_chunk = item
                    yield item
            elif chunk_result:
                current_chunk = chunk_result
                yield chunk_result

        if isinstance(current_chunk, str):
            if len(current_chunk) == 0:
                return
            if not current_chunk[-1] in ".?!'":
                yield "."

    def handle_generator(self, generator):
        for item in generator:
            if isinstance(item, Generator):
                for sub_item in self.handle_generator(item):
                    yield sub_item
            else:
                yield item

    async def handle_message(self):
        try:
            await self.client.send_message("started_handle")
            self.add_to_messages(self.gpt_response)
            await self.client.send_message("added_to_messages")
            functions = self.get_functions(self.gpt_response)
            await self.client.send_message("got_functions")
            res = self.ask_gpt(functions)
            await self.client.send_message("got_gpt_gen")

            current_chunk = ""

            for chunk in res:
                chunk_result = await self.handle_chunk(chunk)
                if isinstance(chunk_result, Generator):
                    for item in self.handle_generator(chunk_result):
                        current_chunk = item
                        yield item
                elif chunk_result:
                    current_chunk = chunk_result
                    yield chunk_result

            if isinstance(current_chunk, str):
                if current_chunk and not current_chunk[-1] in ".?!'":
                    yield "."

        except Exception as e:
            raise e
            yield "I got an error please try again"
            await send_to_websocket("", True)
            return
