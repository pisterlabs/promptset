import tiktoken
import openai
import json
from enum import Enum

class DataPolicy(Enum):
    PURE = "pure"
    REMOVE = "remove"
    SUMMARIZE = "summarize"

class FunctionPolicy(Enum):
    INCLUDE = "include"
    IGNORE = "ignore"

class ContextWindow:
    def __init__(self, high_level_overview, default_data_policy=DataPolicy.PURE, model_name="gpt-3.5-turbo", max_tokens=2048, summarization_system = None):
        """
        Initializes the ContextWindow class.
        
        Args:
            high_level_overview (str): High level explanation of the conversation.
            default_data_policy (str, optional): Default data handling policy. Defaults to "pure".
            model_name (str, optional): Name of the OpenAI model being used. Defaults to "gpt-3.5-turbo".
            max_tokens (int, optional): Maximum tokens for a message. Defaults to 2048.
            summarization_system (str, optional): Specific instructions for how to conduct summarization.
        """
        self.high_level_overview = high_level_overview
        self.high_level_overview_tokens = _token_count(high_level_overview)
        self.default_data_policy = default_data_policy
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.messages = []
        self.enhanced_messages = []
        self.total_tokens = self.high_level_overview_tokens
        self.tokens_sent = 0  # Counter for tokens sent to the OpenAI API

        self.summarization_messages = []
        summarization_system = summarization_system if summarization_system else "You are an assistant that is helping make a more functional version of GPT. In iterative, data-intensive conversations, such as debugging, the context window gets filled with old data. For example, after debugging an error stack, keeping that error stack in the context window on subsequent messages is unproductive. To counter this, you are being used to summarize previous data that is no longer directly relevant, but for which some information must be saved. For example, if the user copy and pastes their entire code, you might return "The user provided their entire program, but to save context window space this has been optimized out of the conversation"."
        self.summarization_messages.append(_construct_role_dict("system", summarization_system))

        self.messages.append(_construct_role_dict("system", self.high_level_overview))

    def add_message(self, user_content, data=None, data_policy=None, function_info=None, function_policy=FunctionPolicy.IGNORE):
        """
        Adds a new user message to the context window.

        Args:
            user_content (str): The natural language message from the user.
            data (str, optional): Associated data. Defaults to None.
            data_policy (str, optional): Data handling policy. If None, uses default. Defaults to None.
            function_info (dict, optional): Information about the associated function. Defaults to None.
            function_policy (str, optional): Policy regarding function information (include/don"t include). Defaults to "include".
        """
        
        # update messages and enhanced messages list
        self.messages.append(_construct_role_dict("user", user_content + data))
        message_struct = _construct_role_dict(user_content, data, data_policy, function_info, function_policy)
        self.enhanced_messages.append(message_struct)

        # convert previous user message into a compressed version, applying data and function policy
        # -3 is used because the array should be layed out as: user_msg, model_response, user_msg. So previous user msg should be third from back
        if (len(self.messages) >= 3):
          prev_user_message = self.enhanced_messages[-3]
          self.total_tokens -= prev_user_message["token_count"]
          self.messages[-3] = _message_dict_to_message(prev_user_message)
          self.total_tokens += self.messages[-3]["token_count"]

        idx = _fit_context_window()
        model_response = _call_openai(self.messages[idx:], message_struct["function_info"])
        res = {}
        if (model_response.get("function_call")):
          parsed_message = _generate_function_description(model_response.get("function_call"))
          res["model_response"] = parsed_message
          res["function_values"] = model_response.get("function_call")
        else:
          res["model_response"] = model_response.get("content")
        
        self.enhanced_messages.append(res)
        self.messages.append(_construct_role_dict("assistant", res["model_response"]))

        return res

    def generate_message_summary(self, data):
        """
        Generates a summary for the given message based on its data policy.

        Args:
            data (string): Index of the message in the enhanced_messages list.
        """
        self.summarization_messages.append(_construct_role_dict("user", f"Please summarize the following: {data}"))
        model_response = _call_openai(self.summarization_messages)
        self.summarization_messages.remove(-1)

        return model_response.get("content")

    def visualize_context_window(self):
        """
        Provides a representation of the current messages for visualization.
        """
        for idx, message in enumerate(self.messages):
            print(f"Message {idx + 1}:")
            print(f"Role: {message["role"]}")
            print(f"Content: {message["content"]}")
            print("------------------------")

    def _token_count(self, message):
        """
        Computes the token count of a given message for the model specified in the class instance.

        Args:
            message (str): The input message for which tokens need to be counted.

        Returns:
            int: The number of tokens in the message.
        """
        encoder = tiktoken.encoding_for_model(self.model_name)
        return len(encoder.encode(message))

    def _fit_context_window(self):
        """
        Updates the messages list based on the current enhanced_messages. Ensures the conversation stays within the token limit.
        Returns the index from which the messages should be considered to fit within the token limit.
        """
        # Count tokens for the most recent user message in enhanced_messages
        recent_user_message_tokens = self.enhanced_messages[-1]["token_count"]

        # If high_level_overview and recent_user_message together exceed max tokens, raise an error
        if self.high_level_overview_tokens + recent_user_message_tokens > self.max_tokens:
            raise ValueError("High level overview and the recent user message together exceed token limit!")

        # Remaining token budget after accounting for high_level_overview and total tokens used so far
        remaining_tokens = self.max_tokens - self.total_tokens - self.high_level_overview_tokens

        # If remaining tokens are negative, we need to figure out from which index we should consider messages
        if remaining_tokens < 0:
            # Loop from the start, and track the first index from which messages can fit within the remaining token limit
            accumulated_tokens = 0
            start_idx = 0  # Default index in case all messages fit
            for idx, message in enumerate(self.messages):
                accumulated_tokens += message["token_count"]

                # Check if the accumulated tokens from this point onward will fit within the remaining token limit
                if accumulated_tokens <= abs(remaining_tokens):
                    start_idx = idx
                    break

            return start_idx

        return 0  # If all messages fit, return the default index

    def _create_function_info(self, name, description, parameter_list):
        """
        Create a function info dictionary for API function definition.

        :param name: str, name of the function.
        :param description: str, description of the function.
        :param parameter_list: list of tuples, where each tuple contains
                              (param_name, param_type, param_description, is_required),
                              and is_required is a boolean value.
        :return: dict, structured function info.
        """
        function_info = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        for param_name, param_type, param_description, is_required in parameter_list:
            param_info = {
                "type": param_type,
                "description": param_description,
            }

            if param_type == "array":
                param_info["items"] = {"type": "string"}

            function_info["parameters"]["properties"][param_name] = param_info

            if is_required:
                function_info["parameters"]["required"].append(param_name)

        return function_info

    def _generate_function_description(self, model_func):
        """
        Generates a description string based on the function call in the model"s response.

        Args:
            function_response (dict): The "function_call" key from the model"s response.

        Returns:
            str: A description of the function call.
        """
        if model_func == None:
          return None

        function_name = model_func.get("name", "")
        arguments = json.loads(model_func.get("arguments", "{}"))
        
        if not arguments:
            return f"The model attempted to call the function "{function_name}" but did not provide any parameters."
        
        description_parts = [f"The model returned function parameters for the requested function "{function_name}"."]
        
        for param, value in arguments.items():
            description_parts.append(f"For parameter "{param}", the model generated "{value}".")
        
        return " ".join(description_parts)

    def _construct_message_dict(self, user_content, data=None, data_policy=None, function_info=None, function_policy=FunctionPolicy.IGNORE):
        """
        Constructs a message dictionary with user content, data, data policy, and function information. 

        Args:
            user_content (str): The main content of the message from the user.
            data (Any, optional): Any associated data with the message. Defaults to None.
            data_policy (str, optional): Policy for how to handle the provided data. Can be "pure", "remove", or another custom policy. Defaults to None.
            function_info (dict, optional): Dictionary containing function name, description, and parameters. Defaults to None.
            function_policy (str, optional): Policy for how to handle the function information. Can be "include", "ignore", or another custom policy. Defaults to "ignore".

        Returns:
            dict: A structured message dictionary with the provided data and policies.
        """
        message_dict = {
            "user_content": user_content,
            "data": data,
            "data_policy": data_policy,
            "function_info": _create_function_info(function_info["name"], function_info["description"], function_info["parameter_list"]),
            "function_policy": function_policy,
            "token_count": _token_count(user_content) + _token_count(data) + _token_count(str(function_info))
        }

        return message_dict

    def _message_dict_to_message(self, message_dict):
        """
        Convert a message dictionary into a string representation.

        Parameters:
        - message_dict: Dictionary containing message data.

        Returns:
        - A string representation of the message.
        """

        final_message = ""
        final_message += message_dict["user_content"]
        message_dict["token_count"] = _token_count(message_dict["user_content"])

        type_map = {
            "int": "a number representing",
            "string": "a string of",
            "boolean": "a boolean indicating",
            "array": "an array of",
        }

        data_msg = ""
        if message_dict["data_policy"] == DataPolicy.PURE:
            data_msg = data
        elif message_dict["data_policy"] == DataPolicy.SUMMARIZE:
            summary = generate_message_summary(data)
            data_msg = summary
        elif message_dict["data_policy"] == DataPolicy.REMOVE:
            data_msg = None
        else:
            raise ValueError("Did not pass a valid function policy!")

        if data_msg != "":
          final_message += f" Along with this message, the user sent the following data: {data_msg}."
          message_dict["token_count"] += _token_count(data_msg)

        if message_dict["function_policy"] == "include" and function_info:
            function_desc = f" Along with this message, the user requested for their function named {function_info["name"]} to be supplied with the following parameters:"
            params = []

            for param_name, param_info in function_info["parameters"]["properties"].items():
                param_type = param_info["type"]
                param_type_desc = type_map.get(param_type, param_type)  # Use the type itself if not in the map

                if param_type == "array":
                    param_type_desc += param_info["items"]["type"] + "s"
                else:
                    param_type_desc += " " + param_info["description"]

                params.append(param_type_desc)

            function_desc += " " + ", ".join(params) + "."
            final_message += function_desc
            message_dict["token_count"] += _token_count(function_desc)

        return final_message

    def _construct_role_dict(self, role, content):
        """
        Constructs a dictionary representing a role and its content.

        Parameters:
        - role: Role string (e.g., "user", "system").
        - content: Content associated with the role.

        Returns:
        - A dictionary with "role" and "content".
        """

        return {
          "role": role,
          "content": content
        }

    def _call_openai(self, messages, function_info=None, max_retries=1, wait_time=1, examples=[]):
        """
        Communicate with the OpenAI API to get a response.

        Parameters:
        - messages: List of message dictionaries to send.
        - function_info: Optional dictionary containing function details.
        - max_retries: Number of times to retry in case of a JSONDecodeError.
        - wait_time: Time in seconds to wait between retries.
        - examples: List of example dictionaries.

        Returns:
        - The response message from OpenAI API.
        """

        self.tokens_sent += self.total_tokens
        functions = [function_info] if function_info else []

        for example in examples:
            example_message = {
                "role": "user",
                "content": f"Document: {example["document"]}. Fields: {json.dumps(example["fields"])}."
            }
            messages.append(example_message)

        retries = 0
        while retries < max_retries:
            try:
                if functions:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        functions=functions,
                        temperature=0
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0
                    )

                response_message = response["choices"][0]["message"]
                return response_message
            except json.JSONDecodeError:
                retries += 1
                time.sleep(wait_time)

        raise Exception("Maximum retries reached. API issues.")