import openai
from agent import Agent
from language_understanding_module import LanguageUnderstandingModule
from probabilistic_reasoning_module import ProbabilisticReasoningModule
from world_state import WorldState
from gui import GUI
from performance_metrics import PerformanceMetrics
import random


class Simulation:
    PERSONALITIES = ["curious", "cautious", "adventurous", "meticulous", "spontaneous"]
    MAX_TICKS = 1000

    def __init__(self, api_key, agent_count):
        # Initialize the API
        openai.api_key = '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'

        # Initialize the Language Understanding Module with the specified model
        model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Model for English to French translation
        openai.api_key = api_key
        self.language_module = LanguageUnderstandingModule(api_key)

        self.reasoning_module = ProbabilisticReasoningModule()

        # Create a dynamic world state
        self.world_state = WorldState()

        # Create agents for the simulation with diverse personalities
        self.agents = [Agent(self.language_module, self.reasoning_module.probabilistic_programming_language, api_key, personality, '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', id) for id, personality in enumerate(self.generate_personalities(agent_count))]
        


        # Create a GUI for user interaction and visualization
        self.gui = GUI()

        # Create a performance metrics object
        self.performance_metrics = PerformanceMetrics()
        self.language_module = LanguageUnderstandingModule(api_key)

        self.ticks = 0
        print(f"Simulation initialized with {agent_count} agents.")

    def generate_personalities(self, count):
        # Generate diverse personalities for agents
        return [random.choice(self.PERSONALITIES) for _ in range(count)]

    def update_simulation_state(self):
        for agent in self.agents:
            # get a list of all other agents
            other_agents = [a for a in self.agents if a is not agent]
        agent.act(other_agents)


    def handle_user_input(self, user_input):
        try:
            # Assume user_input is a string in the format "agent_id:command"
            agent_id, command = user_input.split(":")

            # Find the corresponding agent
            agent = next((a for a in self.agents if a.id == int(agent_id)), None)

            if agent:
                ...
            else:
                return f"Agent with ID {agent_id} not found in the simulation."

        except Exception as e:
            return f"Error while processing user input '{user_input}': {e}"

    def execute(self):
        # The main execution loop of the simulation
        print("Starting simulation...\n")
        
        while not self.termination_condition():
            self.update_simulation_state()

            # Get user input from GUI
            user_input = self.gui.get_user_input()
        if user_input is not None:
            user_result = self.handle_user_input(user_input)
            print(f"User input result: {user_result}")

            # Update GUI
            self.gui.update(self.world_state, self.agents)

            # Update performance metrics
            self.performance_metrics.update(self.agents)

            # Interact with other agents - call the query_other_agents method
            if some_interaction_condition_met:
                self.query_other_agents(querying_agent_id, queried_agent_id)

            # Increment the tick counter
            self.ticks += 1

        print("Simulation terminated.")

    def termination_condition(self):
        # Terminate the simulation if the maximum number of ticks has been reached
        return self.ticks >= self.MAX_TICKS

    def query_other_agents(self, querying_agent_id, queried_agent_id):
        # Find the querying and queried agents
        querying_agent = next(agent for agent in self.agents if agent.id == querying_agent_id)
        queried_agent = next(agent for agent in self.agents if agent.id == queried_agent_id)

        # Get the most recent action of the queried_agent
        recent_action = queried_agent.recent_action

        # Interpret the action using the querying agent's understanding
        interpreted_action = querying_agent.interpret_action(recent_action)

        # Update the querying_agent's beliefs based on the interpreted action
        querying_agent.update_beliefs(interpreted_action, confidence=0.5)
