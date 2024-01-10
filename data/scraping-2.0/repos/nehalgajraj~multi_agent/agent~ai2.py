import json
import openai
from typing import List, Dict, Callable, Any

class AI_generator:
    """
    This class provides the functionality to generate AI responses using the OpenAI API.
    """
    def __init__(self):
        self.available_functions = {}
        self.functions_metadata = []

    def map_functions(self, FUNCTIONS: List[Dict]) -> Dict[str, Callable]:
        """
        Maps function metadata to actual function references.
        """
        return {func_map["metadata"]["name"]: func_map["function"] for func_map in FUNCTIONS}

    def extract_metadata(self, FUNCTIONS: List[Dict]) -> List[Dict]:
        """
        Extracts metadata of the functions.
        """
        return [func_map["metadata"] for func_map in FUNCTIONS]

    def call_function(self, function_name: str, function_args: str) -> Dict[str, Any]:
        """
        Calls the function by the provided name with given arguments.
        """
        try:
            function_to_call = self.available_functions[function_name]
            function_args = json.loads(function_args.strip())
            return function_to_call(**function_args)
        except Exception as e:
            print(f"Error calling function {function_name}: {e}")
            return None

    def generate_response(self, prompt: str, user: str) -> openai.api_resources.abstract.APIResource:
        """
        Generates an AI response using the OpenAI API.
        """
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]

        function_call_count = 0
        while function_call_count < 5:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=messages,
                    functions=self.functions_metadata,
                    function_call="auto",
                )

                response_message = response["choices"][0]["message"]

                if not response_message.get("function_call"):
                    break

                function_response = self.call_function(
                    response_message["function_call"]["name"], 
                    response_message["function_call"]["arguments"]
                )

                messages.append(response_message)
                messages.append(
                    {
                        "role": "function",
                        "name": response_message["function_call"]["name"],
                        "content": json.dumps(function_response),
                    }
                )

                function_call_count += 1
            except Exception as e:
                print(f"Error generating response: {e}")
                break

        return response

    def run(self, FUNCTIONS: List[Dict], prompt: str, user: str):
        """
        Initializes function mappings and generates a response.
        """
        print(f'Running with prompt: "{prompt}" and user input: "{user}"')
        try:
            self.available_functions = self.map_functions(FUNCTIONS)
            self.functions_metadata = self.extract_metadata(FUNCTIONS)
            return self.generate_response(prompt, user)
        except Exception as e:
            print(f"Error in run: {e}")
            return None

    def talk_to_other_agent(self, other_agent, prompt: str, user: str):
        """
        Interacts with another AI agent.
        """
        print(f'Running with prompt: "{prompt}" and user input: "{user}"')
        try:
            self_response = self.generate_response(prompt, user)
            other_agent_response = other_agent.run(prompt, self_response['choices'][0]['message']['content'])
            return other_agent_response
        except Exception as e:
            print(f"Error in talk_to_other_agent: {e}")
            return None

