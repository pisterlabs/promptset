import argparse
from typing import Optional
from langchain.llms.ollama import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from termcolor import colored

class RubberDuck:
    """
    This class is a wrapper around the Ollama model.
    """
    def __init__(self, model: str = "codellama") -> None:
        """
        This function initializes the RubberDuck class.

        Args:
            model (str, optional): The model to be used. Defaults to "codellama".
        """
        self.system_prompt = """You are a pair progamming tool to help developers debug, think through design, and write code. 
        Help the user think through their approach and provide feedback on the code."""
        self.llm = Ollama(model=model, callbacks=[StreamingStdOutCallbackHandler()], system=self.system_prompt)
       

    def call_llama(self, code: str = "", prompt: Optional[str] = None, chain: bool = False) -> None:
        """
        This function calls the Ollama model to provide feedback on the given code.

        Args:
            code (str): The code to be reviewed.
            prompt (Optional[str]): Custom prompt to be used. Defaults to None.
        """
        if prompt is None:
            prompt = "review the code, find any issues if any, suggest cleanups if any:" + code
        else:
            prompt = prompt + code


        self.llm(prompt)
        if chain:
            while(True):
                prompt = input(colored("\n What's on your mind? \n ", 'green'))
                self.llm(prompt)
    

def read_files_from_dir(directory: str) -> str:
    """
    This function reads all the files from a directory and returns the concatenated string.

    Args:
        directory (str): The directory to be processed.

    Returns:
        str: The concatenated string of all the files.
    """
    import os
    files = os.listdir(directory)
    code = ""
    for file in files:
        code += open(directory + "/" + file).read()
    return code


def ducky() -> None:
    """
    This function parses the command line arguments and calls the Ollama model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", help="Custom prompt to be used", default=None)
    parser.add_argument("--file", "-f", help="The file to be processed", default=None)
    parser.add_argument("--directory", "-d", help="The directory to be processed", default=None)
    parser.add_argument("--chain", "-c", help="Chain the output of the previous command to the next command", action="store_true", default=False)
    parser.add_argument("--model", "-m", help="The model to be used", default="codellama")
    args, _ = parser.parse_known_args()

    # My testing has shown that the codellama:7b-python is good for returning python code from the program.
    # My intention with this tool was to give more general feedback and have back a back and forth with the user.
    rubber_ducky = RubberDuck(model=args.model)
    if args.file is None and args.directory is None:
        if args.chain:
            while(True):
                prompt = input(colored("\n What's on your mind? \n ", 'green'))
                rubber_ducky.call_llama(prompt=prompt, chain=args.chain)
        else:
            prompt = input(colored("\n What's on your mind? \n ", 'green'))
            rubber_ducky.call_llama(prompt=prompt, chain=args.chain) 

    if args.file is not None:
        code = open(args.file).read()
        rubber_ducky.call_llama(code=code, prompt=args.prompt, chain=args.chain)
    
    elif args.directory is not None:
        code = read_files_from_dir(args.directory)
        rubber_ducky.call_llama(code=code, prompt=args.prompt, chain=args.chain)
    

if __name__ == "__main__":
    ducky()
