from langchain.utilities import SerpAPIWrapper
from langchain.agents import tool
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent

def PersonalRecommenderAgent():
    """
    Agent that acts as a recommender based on user inputs and conversation.
    """

    def __init__(self):
        """
        Initializes the recommender agent.
        """
        self.llm = OpenAI(temperature=0)
        self.history = [] # list of strings

    def get_action_via_chat_completion(self, prompt: str, max_retries = 2):
        """
        Get the action to take.
        """
        tools = load_tools(["serpapi", "llm-math"], llm=self.llm) # potentially add more tools here
        agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)
        response = agent.run(prompt)
        return response


    def select_action(self, user_interaction, possible_actions: dict):
        """
        Select an action from the action space, based on the conversation with the user, and the response to the recommendations.
        """

        action_prompt = f"""
            You are given the following interaction with a user looking for clothes to buy:\n\n
            {user_interaction}\n\n
            The following is a list of actions you can take now, represented as a Python dictionary, :\n\n
            {possible_actions}\n\n
            Read the actions carefully, and pick *one* action that you would take, 
            which is most likely to satisfy the user based on what you know about them from the conversation. Return no other text, only the key to the dictionary that corresponds to the action you would like to take.
            """

        action_str = self.get_action_via_chat_completion(action_prompt)

        print(f"Action selected: {action_str}")

        return action_str

