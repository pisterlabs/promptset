import os
import json
import openai
from django.db import connection
from mixr.constants import TOOLS
from termcolor import colored


class AiInterface:
    def __init__(self):
        self.SEED = 42

        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def add_raw_message(self, message):
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )
    
    # Chat completion requests
    def chat_completion_request(self, messages, tools=None, tool_choice=None):
        # TODO: System messageai
        # You are a professional bartender skilled in the art of making cocktails. You are chatting with a customer who wants to make a cocktail. The customer says:
        
        return self.client.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            seed=self.SEED,
            tools=TOOLS,
        )
    
    def chat_completion_with_function_execution(self):
        response = self.chat_completion_request(self.conversation_history)
        print(response)
        print("----------------")

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            self.add_raw_message(response_message)
            for tool_call in tool_calls:
                results = self.execute_function_call(response_message)
                self.add_raw_message({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": results})
            self.chat_completion_with_function_execution()
        else:
            self.add_message("assistant", response.choices[0].message.content)


    def send_message(self):
        self.conversation_history.append({"role": "system", "content": "Answer the user's question about cocktails by querying the Django database."})
        self.conversation_history.append({"role": "user", "content": "What are some easy cocktails I can make?"})

        self.chat_completion_with_function_execution()
        self.pretty_print_conversation(self.conversation_history)
        self.conversation_history = []

    # Database functions
    def ask_database(self, query):
        with connection.cursor() as cursor:
            cursor.execute(query)
            try:
                rows = cursor.fetchall()
                results = str(rows)
            except Exception as e:
                results = f"query failed with error: {e}"

        return results

    def execute_function_call(self, message):
        function_name = message.tool_calls[0].function.name
        if function_name == "ask_database":
            query = message.tool_calls[0].function.arguments
            query = json.loads(query)["query"]
            print("QUERY: ", query)
            results = self.ask_database(query)
        else:
            results = f"Error: function {function_name} does not exist"
        return results


    def pretty_print_conversation(self, messages):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "tool": "magenta",
        }
        
        for message in messages:
            try:
                role = message["role"]
                content = message["content"]
                function_call = message.get("function_call")
            except TypeError:
                role = message.role
                content = message.content
                function_call = message.tool_calls[0].function.name

            if role == "system":
                print(colored(f"system: {content}\n", role_to_color[role]))
            elif role == "user":
                print(colored(f"user: {content}\n", role_to_color[role]))
            elif role == "assistant" and function_call:
                print(colored(f"assistant: {function_call}\n", role_to_color[role]))
            elif role == "assistant" and not function_call:
                print(colored(f"assistant: {content}\n", role_to_color[role]))
            elif role == "tool":
                print(colored(f"function ({message['name']}): {content}\n", role_to_color[role]))

