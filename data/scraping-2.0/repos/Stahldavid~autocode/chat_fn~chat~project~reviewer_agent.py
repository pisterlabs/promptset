# Import langchain modules
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentActionOutputParser
from langchain.tools import CodeReviewerTool

# Import language models
from openai_request_llm import OpenAI # Custom LLM based on OpenAI

# Define prompt template
prompt_template = """
This is a reviewer agent that can perform different actions related to code reviewing and improvement suggestions using CodeReviewer tool.
You can ask me to review any code and specify the language you want.
To specify the language, use the syntax: #language: code
For example:
#python: def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
#javascript: function reverseString(str) {
    return str.split("").reverse().join("");
}
#c++: int sum(int a, int b) {
    return a + b;
}

You can also chat with me about code quality, best practices, or anything else.
For example:
How can I improve my coding skills?
What are some common coding mistakes to avoid?
Hello, how are you today?
"""

# Choose language model
language_model = OpenAI()

# Define stop sequence
stop_sequence = "\n"

# Define output parser
output_parser = AgentActionOutputParser()

# Load tools
codereviewer_tool = CodeReviewerTool()

# Create custom agent class by subclassing BaseMultiActionAgent
class ReviewerAgent(BaseMultiActionAgent):
    def __init__(self, prompt_template, language_model, stop_sequence, output_parser):
        super().__init__(prompt_template, language_model, stop_sequence, output_parser)

    def decide_action(self, user_input):
        # Override this method to decide which action to take based on user input
        # You can use any logic or condition you want
        # Return an action name and an action input

        # If user input starts with #language:, use codereviewer tool to review code and provide feedback or suggestions for that language and code
        if user_input.startswith("#"):
            action_name = "codereviewer"
            action_input = user_input.strip()
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

        # If action name is codereviewer, use codereviewer tool to review code and provide feedback or suggestions for the given language and code
        if action_name == "codereviewer":
            output = codereviewer_tool.run(action_input)

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