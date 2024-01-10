import ast
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Callable, Type, Tuple

import tiktoken
from openai.types.chat import ChatCompletionToolParam

from nimbusagent.functions import parser
from nimbusagent.functions.responses import FuncResponse, DictFuncResponse
from nimbusagent.memory.base import AgentMemory
from nimbusagent.utils.helper import find_similar_embedding_list, combine_lists_unique


@dataclass
class FunctionInfo:
    """
    Class that stores information about a function.
    """
    name: str
    definition: str
    mapping: Union[Callable, Any]
    mapping_name: str
    tokens: int


class FunctionHandler:
    """
    Class that handles function calls.  This class is responsible for parsing functions, creating function mappings,
    and calling functions. It also handles the logic for determining which functions to use based on the query and
    chat history.
    :param functions:  The list of functions to use.  If None, the functions will be parsed from the function_handler.
    :param embeddings:  The list of function embeddings to use.  If None, the functions will be parsed from the
                            function_handler.
    :param k_nearest:  The number of nearest neighbors to use when finding similar functions.  Defaults to 3.
    :param always_use:  The list of functions to always use.  If None, no functions will be used by default.
    :param pattern_groups:  The list of pattern groups to use.  If None, no pattern groups will be used.
    :param calling_function_start_callback:  The callback to call when a function is called.  If None, no callback
                            will be called.
    :param calling_function_stop_callback:  The callback to call when a function is finished being called.  If None,
                            no callback will be called.
    :param chat_history:  The chat history to use.  If None, no chat history will be used.
    """
    functions = None
    func_mapping = None
    always_use = None
    orig_functions: dict = None
    pattern_groups = None
    chat_history: AgentMemory = None
    processed_functions = None
    max_tokens = 0

    def __init__(self, functions: list = None,
                 embeddings: list = None,
                 k_nearest: int = 3,
                 min_similarity: float = 0.5,
                 always_use: list = None,
                 pattern_groups: list = None,
                 calling_function_start_callback: callable = None,
                 calling_function_stop_callback: callable = None,
                 chat_history: AgentMemory = None,
                 max_tokens: int = 0
                 ):

        self.embeddings = embeddings
        self.k_nearest = k_nearest
        self.min_similarity = min_similarity
        self.always_use = always_use
        self.pattern_groups = pattern_groups
        self.chat_history = chat_history
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

        self.orig_functions = {func.__name__: func for func in functions} if functions else None
        if not embeddings:
            self.functions = self.parse_functions(functions)
            self.func_mapping = self.create_func_mapping(functions)

        self.calling_function_start_callback = calling_function_start_callback
        self.calling_function_stop_callback = calling_function_stop_callback

    def functions_to_tools(self) -> List[ChatCompletionToolParam]:
        """
        Convert the functions defs to the new OpenAI tools format.
        :return:  The tools.
        """
        tools = []
        for func in self.functions:
            tools.append({
                "type": "function",
                "function": func
            })

        return tools

    def _get_function_info(self, func_name: str) -> Optional[FunctionInfo]:
        """
        Get the FunctionInfo for the given function name.
        :param func_name:  The name of the function to get the FunctionInfo for.
        :return:  The FunctionInfo for the given function name. If the function name is not found, None is returned.
        """

        if not self.orig_functions:
            return None

        func = self.orig_functions.get(func_name)
        if not func:
            return None

        func_definition = parser.func_metadata(func)
        func_tokens = self.tokenize(json.dumps(func_definition))
        mapping_name, mapping = self.create_individual_func_mapping(func)
        return FunctionInfo(name=func_name,
                            definition=func_definition,
                            tokens=func_tokens,
                            mapping=mapping,
                            mapping_name=mapping_name)

    def _get_group_function(self, query: str) -> Optional[List[str]]:
        """
        Get the list of functions to use based on the pattern groups.
        :param query:  The query to use.
        :return:  The list of functions to use based on the pattern groups. If no pattern groups are found, None is
                    returned.
        """
        if not self.pattern_groups:
            return None

        function_names = []

        for group in self.pattern_groups:
            if re.search(group['pattern'], query):
                for func in group['functions']:
                    func_name = func if isinstance(func, str) else func.__name__
                    if func_name not in function_names and func_name in self.orig_functions:
                        function_names.append(func_name)

        return function_names

    def remove_functions_mappings(self, function_names: List[str]):
        """
        Remove the given functions from the function mappings.
        :param function_names:  The list of function names to remove.
        """
        if not self.processed_functions:
            return

        use_functions = []
        use_names = []
        token_count = 0
        for func in self.processed_functions:
            if func.name not in function_names:
                use_functions.append(func)
                use_names.append(func.name)
                token_count += func.tokens

        self._set_functions_and_mappings(use_functions)
        logging.info("Removed functions: %s", function_names)
        logging.info(f"Using functions: {use_names}")
        logging.info(f"Total tokens: {token_count}")

    def reset_functions_mappings(self):
        """
        Reset the function mappings to the original functions.
        """
        self._set_functions_and_mappings(self.processed_functions)

    def _set_functions_and_mappings(self, functions: Optional[List[FunctionInfo]] = None):
        """
        Set the functions and function mappings to the given functions.
        :param functions:  The list of functions to use.  If None, the functions will be parsed from
                    the function_handler.
        """
        if functions:
            self.functions = [parser.func_metadata(func.mapping) for func in functions]
            self.func_mapping = {func.mapping_name: func.mapping for func in functions}
        else:
            self.functions = None
            self.func_mapping = None

    def get_functions_from_query_and_history(self, query: str, history: List[Dict[str, Any]]):
        """
        Get the functions to use based on the query and history.
        :param query:  The query to use.
        :param history:  The history to use. A list of dictionaries with 'role' and 'content' fields.
        """
        if not self.orig_functions:
            return None

        if not self.pattern_groups and not self.embeddings:
            actual_function_names = self.orig_functions.keys()

        else:
            # Step 1: Initialize with 'always_use' functions
            actual_function_names = self.always_use if self.always_use else []
            # print("actual_function_names: ", actual_function_names)

            # step 2: Add functions based on pattern groups on query
            query_group_functions = self._get_group_function(query)
            if query_group_functions:
                actual_function_names = combine_lists_unique(actual_function_names, query_group_functions)

            # step 3: Add functions based on embeddings
            recent_history_and_query = [message['content'] for message in history[-2:]] + [query]
            recent_history_and_query_str = " ".join(recent_history_and_query)

            if self.embeddings:
                similar_functions = find_similar_embedding_list(recent_history_and_query_str,
                                                                function_embeddings=self.embeddings,
                                                                k_nearest_neighbors=self.k_nearest)
                similar_function_names = [d['name'] for d in similar_functions]
                if similar_function_names:
                    actual_function_names = combine_lists_unique(actual_function_names, similar_function_names)

            # step 4: Add functions based on pattern groups on history
            query_group_functions = self._get_group_function(recent_history_and_query_str)
            if query_group_functions:
                actual_function_names = combine_lists_unique(actual_function_names, query_group_functions)

            logging.info(f"Actual Functions Names to use: {actual_function_names}")
            # step 5: step through functions and get the function info, adding up to max_tokens

        processed_functions = []
        token_count = 0
        for func_name in actual_function_names:
            func_info = self._get_function_info(func_name)
            if func_info:
                processed_functions.append(func_info)
                token_count += func_info.tokens
                if 0 < self.max_tokens <= token_count:
                    break

        self.processed_functions = processed_functions
        using_functions = [func.name for func in processed_functions]
        logging.info(f"query: {query}")
        logging.info(f"Using functions: {using_functions}")
        logging.info(f"Total tokens: {token_count}")

        # step 6: update self.functions and self.func_mapping
        self._set_functions_and_mappings(processed_functions)

    @staticmethod
    def parse_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse the messages to get the role and content fields.  If the messages do not have the role and content fields,
                they will be ignored.
        :param messages: The messages to parse. A list of dictionaries with 'role' and 'content' fields.
        :return:  The parsed messages. A list of dictionaries with 'role' and 'content' fields.
        """
        return [
            {"role": message['role'], "content": message['content']}
            for message in messages
            if 'role' in message and 'content' in message
        ]

    @staticmethod
    def parse_functions(functions: List[Union[Callable, Type]]) -> List[Dict[str, Any]]:
        """
        Parse the functions to get the function metadata.
        :param functions:  The functions to parse. If None, the functions will be parsed from the function_handler.
        :return:  The parsed functions. A list of dictionaries with 'name', 'parameters', and 'return' fields.
        """
        if not functions:
            return []
        return [parser.func_metadata(func) for func in functions]

    def create_func_mapping(self, items: List[Union[Callable, Type]]) -> Dict[str, Union[Callable, Any]]:
        """
        Create a function mapping from the given items.  The function mapping is a dictionary with the function name as
                the key and the function as the value.
        :param items:  The items to create the function mapping from.  If None, the function mapping will be created
                       from the function_handler.
        :return:  The function mapping. A dictionary with the function name as the key and the function as the value.
        """
        if not items:
            return {}

        mapping = {}
        for item in items:
            key, value = self.create_individual_func_mapping(item)
            mapping[key] = value

        return mapping

    @staticmethod
    def create_individual_func_mapping(item: Union[Callable, Type]) -> Tuple[str, Callable]:
        """
        Create a function mapping from the given item.  The function mapping is a tuple with the function name as
                the first element and the function as the second element.
        :param item:  The item to create the function mapping from.
        :return:  The function mapping. A tuple with the function name as the first element and the function as the
                    second element.
        """
        if callable(item):  # It's a function
            return item.__name__, item
        elif inspect.isclass(item):  # It's a class
            return item.__name__, item()  # Create an instance of the class
        else:
            raise ValueError(f"Unsupported item {item}")

    def handle_function_call(self, func_name: str, args_str: str) -> Optional[FuncResponse]:
        """
        Handle a function call.  This method will call the function and return the result.  If the result is a
                FuncResponse, it will be returned as is.  If the result is a dictionary, it will be converted to a
                DictFuncResponse and returned.  If the result is None, None will be returned.
        :param func_name:  The name of the function to call.
        :param args_str:  The arguments to pass to the function. The arguments are a JSON formatted string.
        :return:  The result of the function call.  If the result is a FuncResponse, it will be returned as is.  If
                    the result is a dictionary, it will be converted to a DictFuncResponse and returned.  If the
                    result is None, None will be returned.
        """
        result = self._call_function(func_name, args_str)

        if result is None:
            return None

        # Map the result to the appropriate AbstractFuncResponse subclass
        if isinstance(result, FuncResponse):
            response_obj = result
        else:
            response_obj = DictFuncResponse(result)

        response_obj.name = func_name
        response_obj.arguments = args_str

        return response_obj

    @staticmethod
    def _execute_method(item: Any, method_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute the given method on the given item with the given arguments.  If the item is a class type, an instance
                of the class will be created and the method will be executed on the instance.  If the item is a class
                instance, the method will be executed on the instance.
        :param item:  The item to execute the method on.
        :param method_name:     The name of the method to execute.
        :param args:  The arguments to pass to the method.
        :return:  The result of the method call.
        """
        method = getattr(item, method_name)
        if not callable(method):
            raise ValueError(f"Object {item} does not have a callable '{method_name}' method.")

        args.pop('return', None)  # Remove the 'return' argument if it exists
        return method(**args)

    def _call_function(self, func_name: str, args_str: str):
        args = self.get_args(args_str)

        if self.calling_function_start_callback:
            self.calling_function_start_callback(func_name, args)

        item = self.func_mapping.get(func_name)
        if item is None:
            raise ValueError(f"Function or class {func_name} not found in function mapping.")

        if callable(item) and not inspect.isclass(item):  # It's a function
            res = item(**args)
        else:  # It's a class or class instance
            method_name = getattr(item, 'method_name', 'call')
            if inspect.isclass(item):  # It's a class type
                instance = item()
                if hasattr(item, 'set_chat_history'):
                    method = getattr(item, 'set_chat_history')
                    method(instance, self.chat_history)
                res = self._execute_method(instance, method_name, args)
            else:  # It's a class instance
                if hasattr(item, 'set_chat_history'):
                    method = getattr(item, 'set_chat_history')
                    method(item, self.chat_history)
                res = self._execute_method(item, method_name, args)

        if self.calling_function_stop_callback:
            self.calling_function_stop_callback()
        return res

    @staticmethod
    def get_args(args_str: str):
        """
        Get the arguments from the given JSON formatted string.
        :param args_str:  The JSON formatted string to get the arguments from.
        :return:  The arguments as a dictionary.
        """
        return ast.literal_eval(args_str)

    @property
    def functions_list(self) -> List[Dict[str, Any]]:
        """
        Get the list of functions.
        :return:  The list of functions. A list of dictionaries with 'name', 'parameters', and 'return' fields.
        """
        return self.functions

    def tokenize(self, content: str) -> int:
        """
        Tokenize the content and return the number of tokens.
        :param content:  The content to tokenize.
        :return:  The number of tokens.
        """
        return len(self.encoding.encode(content))
