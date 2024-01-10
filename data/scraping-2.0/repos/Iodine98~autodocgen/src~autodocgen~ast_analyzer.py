import ast
import logging
import os
import re
from _ast import AST
from pathlib import Path
from typing import Union, Optional, Literal, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from . import FileVisitor
    from . import DocGenDef

import black
import dotenv
import openai

dotenv.load_dotenv()
Role = Union[Literal["user"], Literal["system"], Literal["assistant"]]


class ChatMessage(TypedDict):
    """
    ChatMessage class for type hints in OpenAI

    Attributes:
    -----------
    role : Role
        The role of the chat message sender.
    content : str
        The content of the chat message.
    """

    role: Role
    content: str


class ModelKwargs(TypedDict):
    """
    ModelKwargs class for type hints in OpenAI

    Attributes:
    -----------
    max_tokens : int
        The maximum number of tokens to generate in the response.
    model : str
        The name of the model to use for generating the response.
    messages : list[ChatMessage]
        A list of chat messages to use as context for generating the response.
    n : Optional[int]
        The number of responses to generate.
    stop : Optional[Union[str, list]]
        A string or list of strings that, when encountered in the generated response,
        will cause the response to end.
    temperature : int
        The "creativity" of the generated response,
        with higher values resulting in more diverse responses.
    """

    max_tokens: int
    model: str
    messages: list[ChatMessage]
    n: Optional[int]
    stop: Optional[Union[str, list]]
    temperature: int


class ASTAnalyzer:
    """
    ASTAnalyzer class for analyzing and modifying Python ASTs.

    Attributes:
    -----------
    default_messages : list[ChatMessage]
        The default list of chat messages to use as context for generating responses.
    default_model_kwargs : ModelKwargs
        The default model kwargs to use for generating responses.
    default_prepend_prompt : str
        The default prompt to prepend to new prompts.

    Methods:
    --------
    __init__(self, model_kwargs: ModelKwargs=None, prepend_prompt: str=default_prepend_prompt,
     **kwargs)
        Initializes an instance of the ASTAnalyzer class.
    load_ast_from_file(self, file_path: Union[str, Path])
        Loads a Python AST from a file.
    add_new_prompt_to_messages(self, new_prompt: str) -> None
        Adds a new prompt to the list of chat messages.
    add_response_to_messages(self, role: Role, content: str)
        Adds a new response to the list of chat messages.
    update_token_usage(self, new_usage: int)
        Updates the total token usage of the OpenAI API.
    obtain_pydoc(self, new_prompt: str) -> str
        Generates PyDoc for a given prompt.
    write_file_from_ast(self, file_path: Union[str, Path], str_return=False) -> Optional[str]
        Writes a modified Python AST to a file.
    add_docstring_to_ast(node: DocGenDef, new_docstring: str)
        Adds a docstring to a given node in the Python AST.
    remove_messages(self)
        Removes all chat messages except for the default message.
    generate_documentation(self, file_visitor: 'FileVisitor')
        Generates documentation for a Python AST.
    """

    default_messages: list[ChatMessage] = [
        ChatMessage(
            role="user",
            content="Act as a Python software engineer that needs to write Python Docstrings for their modules,"
                    " classes and methods.\n"
                    "The Docstrings should be formatted as Google Style Python Docstrings.\n"
                    "The example will follow.",
        )
    ]
    default_model_kwargs = ModelKwargs(
        max_tokens=1024,
        model="gpt-3.5-turbo",
        messages=default_messages,
        n=1,
        stop=None,
        temperature=0,
    )
    default_prepend_prompt = "Write Pydoc for the function below and only return the PyDoc:\n\n"

    def __init__(
            self,
            model_kwargs: ModelKwargs = None,
            prepend_prompt: str = default_prepend_prompt,
            **kwargs
    ):
        """
        Initializes an instance of the class.

        Args:
            model_kwargs (ModelKwargs, optional): The model's keyword arguments. Defaults to None.
            prepend_prompt (str, optional): The prompt to prepend to the input text.
            Defaults to default_prepend_prompt.
            **kwargs: Additional keyword arguments.

        Raises:
            EnvironmentError: If OPENAI_KEY is missing.

        Returns:
            None
        """
        self.model_kwargs: ModelKwargs = model_kwargs or self.default_model_kwargs
        self.prepend_prompt = prepend_prompt
        self.source_code: Optional[str] = None
        self.tree: Optional[AST] = None
        self.latest_response = None
        self.total_token_usage = 0
        openai.api_key = os.getenv("OPENAI_KEY", None)
        if openai.api_key is None:
            raise EnvironmentError("OPENAI_KEY is missing")
        if "file_path" in kwargs:
            self.load_ast_from_file(kwargs["file_path"])
        if "line_length" in kwargs:
            self.line_length = kwargs["line_length"]
        else:
            self.line_length = 100

    def load_ast_from_file(self, file_path: Union[str, Path]):
        """
        Loads the abstract syntax tree (AST) from a file.

        Args:
            file_path (Union[str, Path]): The path to the file.

        Returns:
            None
        """
        with open(file_path, "r", encoding="utf-8") as file:
            self.source_code = file.read()
        self.tree: AST = ast.parse(self.source_code)

    def add_new_prompt_to_messages(self, new_prompt: str) -> None:
        """
        Adds a new prompt to the messages.

        Args:
            new_prompt (str): The new prompt to add.

        Returns:
            None
        """
        self.model_kwargs["messages"].append(
            ChatMessage(role="user", content=self.prepend_prompt + new_prompt)
        )

    def add_response_to_messages(self, role: Role, content: str):
        """
        Adds a response to the messages.

        Args:
            role (Role): The role of the message.
            content (str): The content of the message.

        Returns:
            None
        """
        self.model_kwargs["messages"].append(ChatMessage(role=role, content=content))

    def update_token_usage(self, new_usage: int):
        """
        Updates the total token usage.

        Args:
            new_usage (int): The new token usage to add.

        Returns:
            None
        """
        self.total_token_usage += new_usage

    def obtain_pydoc(self, new_prompt: str) -> str:
        """
        Obtains the PyDoc for a given prompt.

        Args:
            new_prompt (str): The prompt to obtain the PyDoc for.

        Returns:
            str: The PyDoc for the given prompt.
        """
        self.add_new_prompt_to_messages(new_prompt)
        self.latest_response = openai.ChatCompletion.create(**self.model_kwargs)
        latest_message = self.latest_response.choices[0].message
        self.add_response_to_messages(role=latest_message.role, content=latest_message.content)
        self.update_token_usage(self.latest_response.usage["total_tokens"])
        return latest_message.content.strip()

    def write_file_from_ast(self, file_path: Union[str, Path], str_return=False) -> Optional[str]:
        """
        Writes the AST to a file.

        Args:
            file_path (Union[str, Path]): The path to the file.
            str_return (bool, optional): Whether to return the new code as a string.
            Defaults to False.

        Returns:
            Optional[str]: The new code as a string if str_return is True, otherwise None.
        """
        new_code: str = ast.unparse(ast_obj=self.tree)
        new_code: str = black.format_str(new_code, mode=black.Mode(line_length=self.line_length))
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(new_code)
        if str_return:
            return new_code
        return None

    @staticmethod
    def add_docstring_to_ast(node: "DocGenDef", new_docstring: str):
        """
        Adds a docstring to an abstract syntax tree (AST) node.

        Args:
            node (DocGenDef): The AST node to add the docstring to.
            new_docstring (str): The new docstring to add.

        Returns:
            None
        """
        existing_docstring = ast.get_docstring(node)
        double_quote_match = re.search(r"\"\"\"([\s\S]*?)\"\"\"", new_docstring)
        single_quote_match = re.search(r"'''([\s\S]*?)'''", new_docstring)
        backtick_match = re.search(r"```([\s\S]*?)```", new_docstring)

        if double_quote_match:
            clean_docstring = double_quote_match.group(1)
            logging.debug("Docstring was extracted using three double quotes")
        elif backtick_match:
            clean_docstring = backtick_match.group(1)
            logging.debug("Docstring was extracted using three backticks")
        elif single_quote_match:
            clean_docstring = single_quote_match.group(1)
            logging.debug("Docstring was extracted using three single quotes")
        else:
            clean_docstring = new_docstring
            logging.debug("The docstring could not be extracted.")
        docstring_node = ast.Expr(value=ast.Str(s=clean_docstring + "\n"))
        if existing_docstring:
            node.body[0] = docstring_node
        else:
            node.body.insert(0, docstring_node)

    def remove_messages(self):
        """
        Removes all messages except the first one.

        Returns:
            None
        """
        self.model_kwargs["messages"] = [self.model_kwargs["messages"][0]]

    def generate_documentation(self, file_visitor: "FileVisitor"):
        """
        Generates documentation for the abstract syntax tree (AST).

        Args:
            file_visitor ('FileVisitor'): The file visitor to use.

        Returns:
            None
        """
        file_visitor.visit(self.tree)
