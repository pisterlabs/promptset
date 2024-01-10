# Import langchain modules
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentActionOutputParser
from langchain.tools import SerpAPIWrapper
from langchain.tools import DocstoreExplorer

# Import language models
from openai_request_llm import OpenAI # Custom LLM based on OpenAI

# Define prompt template
prompt_template = """
This is an approach agent that can perform different actions related to finding the best approach for solving a given problem.
You can ask me to solve any problem and I will try to answer or delegate the task to another tool.
To use standard search, use the syntax: #serp_api: query
To use document store explorer, use the syntax: #docstore_explorer: query
To chat with me, just type anything else.
For example:
#serp_api: How to implement a binary search algorithm in Python?
#docstore_explorer: What are some applications of natural language processing?
How do you solve a sudoku puzzle?
"""

# Choose language model
language_model = OpenAI()

# Define stop sequence
stop_sequence = "\n"

# Define output parser
output_parser = AgentActionOutputParser()

# Load tools
serp_api = SerpAPIWrapper()
docstore_explorer = DocstoreExplorer()

# Create custom agent class by subclassing BaseMultiActionAgent
class ApproachAgent(BaseMultiActionAgent):
    def __init__(self, prompt_template, language_model, stop_sequence, output_parser):
        super().__init__(prompt_template, language_model, stop_sequence, output_parser)

    def decide_action(self, user_input):
        # Override this method to decide which action to take based on user input
        # You can use any logic or condition you want
        # Return an action name and an action input

        # If user input starts with #serp_api:, use standard search tool
        if user_input.startswith("#serp_api:"):
            action_name = "serp_api"
            action_input = user_input.replace("#serp_api:", "").strip()
            return action_name, action_input

        # If user input starts with #docstore_explorer:, use document store explorer tool
        elif user_input.startswith("#docstore_explorer:"):
            action_name = "docstore_explorer"
            action_input = user_input.replace("#docstore_explorer:", "").strip()
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

        # If action name is serp_api, use serp_api tool to perform standard search
        if action_name == "serp_api":
            output = serp_api.run(action_input)

        # If action name is docstore_explorer, use docstore_explorer tool to perform document store exploration
        elif action_name == "docstore_explorer":
            output = docstore_explorer.run(action_input)

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
