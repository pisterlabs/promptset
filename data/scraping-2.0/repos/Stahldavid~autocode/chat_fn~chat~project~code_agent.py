# Import langchain modules
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentActionOutputParser
from langchain.tools import GraphCodeBERTTool
from langchain.tools import UnixcoderTool
from langchain.tools import CodeReviewerTool

# Import language models
from openai_request_llm import OpenAI # Custom LLM based on OpenAI

# Define prompt template
prompt_template = """
This is a code agent that can perform different actions related to generating and understanding code using GraphCodeBERT, Unixcoder, and other code-related tools.
You can ask me to generate code for any task and specify the language you want.
To specify the language, use the syntax: #language: task
For example:
#python: write a function that returns the factorial of a number
#javascript: reverse a string
#c++: write a function that returns the sum of two numbers

You can also ask me to understand code and translate it to natural language by using the syntax: understand #language: code
For example:
understand #python: def fib(n):
    if n < 2:
        return n
    else:
        return fib(n-1) + fib(n-2)

You can also ask me to review code and provide feedback or suggestions by using the syntax: review #language: code
For example:
review #python: def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

# Choose language model
language_model = OpenAI()

# Define stop sequence
stop_sequence = "\n"

# Define output parser
output_parser = AgentActionOutputParser()

# Load tools
graphcodebert_tool = GraphCodeBERTTool()
unixcoder_tool = UnixcoderTool()
codereviewer_tool = CodeReviewerTool()

# Create custom agent class by subclassing BaseMultiActionAgent
class CodeAgent(BaseMultiActionAgent):
    def __init__(self, prompt_template, language_model, stop_sequence, output_parser):
        super().__init__(prompt_template, language_model, stop_sequence, output_parser)

    def decide_action(self, user_input):
        # Override this method to decide which action to take based on user input
        # You can use any logic or condition you want
        # Return an action name and an action input

        # If user input starts with #language:, use graphcodebert tool to generate code for that language and task
        if user_input.startswith("#"):
            action_name = "graphcodebert"
            action_input = user_input.strip()
            return action_name, action_input

        # If user input starts with understand #language:, use unixcoder tool to understand code and translate it to natural language
        elif user_input.startswith("understand #"):
            action_name = "unixcoder"
            action_input = user_input.replace("understand ", "").strip()
            return action_name, action_input

        # If user input starts with review #language:, use codereviewer tool to review code and provide feedback or suggestions
        elif user_input.startswith("review #"):
            action_name = "codereviewer"
            action_input = user_input.replace("review ", "").strip()
            return action_name, action_input

        # Otherwise, chat with user using language model
        else:
            action_name = "chat"
            action_input = user_input.strip()
            return action_name, action_input

    def execute_action(self, action_name, action_input):
        # Override this method to execute the action using the appropriate tool or language model
        # You can use any logic or condition you want
        # Return an output string

        # If action name is graphcodebert, use graphcodebert tool to generate code for the given language and task
        if action_name == "graphcodebert":
            output = graphcodebert_tool.run(action_input)

        # If action name is unixcoder, use unixcoder tool to understand code and translate it to natural language
        elif action_name == "unixcoder":
            output = unixcoder_tool.run(action_input)

        # If action name is codereviewer, use codereviewer tool to review code and provide feedback or suggestions
        elif action_name == "codereviewer":
            output = self.codereviewer_tool.run(action_input)

        # If action name is chat, use language model to generate a chat response
        else:
            output = self.language_model.generate(self.prompt_template + "\n" + self.stop_sequence + "\n" + "User: " + action_input + "\n" + "Agent:", stop=self.stop_sequence)

        # Return the output string
        return output

    def parse_output(self, output):
        # Override this method to parse the output using the output parser
        # You can use any logic or condition you want
        # Return a parsed output object

        # Use the output parser to parse the output string into an object with attributes such as text and type
        parsed_output = self.output_parser.parse(output)

        # Return the parsed output object
        return parsed_output