# Import langchain modules
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentActionOutputParser
from langchain.tools import SerpAPIWrapper
from langchain.tools import RetrievalQA


# Define prompt template
prompt_template = """
This is a research agent that can perform different actions related to research tasks, such as searching and semantic searching.
You can ask me anything and I will try to answer or delegate the task to another tool.
To use standard search, use the syntax: #serp_api: query
To use semantic search, use the syntax: #retrieval_qa: query
To chat with me, just type anything else.
For example:
#serp_api: How many planets are there in the solar system?
#retrieval_qa: What is natural language processing?
Hello, how are you today?
"""

# Choose language model
language_model = OpenAI()

# Define stop sequence
stop_sequence = "\n"

# Define output parser
output_parser = AgentActionOutputParser()

# Load tools
serp_api = SerpAPIWrapper()
retrieval_qa = RetrievalQA()

# Create custom agent class by subclassing BaseMultiActionAgent
class ResearchAgent(BaseMultiActionAgent):
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

        # If user input starts with #retrieval_qa:, use semantic search tool
        elif user_input.startswith("#retrieval_qa:"):
            action_name = "retrieval_qa"
            action_input = user_input.replace("#retrieval_qa:", "").strip()
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

        # If action name is retrieval_qa, use retrieval_qa tool to perform semantic search
        elif action_name == "retrieval_qa":
            output = retrieval_qa.run(action_input)

        # If action name is chat, use language model to generate a chat response
        else:
            output = self.language_model.generate(
                self.prompt_template
                + "\n"
                + self.stop_sequence
                + "\n"
                + "User: "
                + action_input
                + "\n"
                + "Agent:",
                stop=self.stop_sequence,
            )

       
