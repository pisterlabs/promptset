""" This class defines a simple agent that uses function calling SDK from OpenAI to avoid the automatic action process of the current Assistants API
"""

from MLAgentBench_v2.agents.agent import Agent
import time
import json
from MLAgentBench_v2.LLM import complete_text_openai

class SingleOldFunctionCallingAgent(Agent):
    """ Agent that uses function calling based on a prompt."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Single Old Function Calling Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant.'''

        MAX_STEPS = 10
        count = 0
        while True:
            # Create the prompt for function calling
            formatted_answer_states = ""
            for idx, answer_state in enumerate(self.answer_states):
                formatted_answer_states += "\nStep: " + str(idx) 
                formatted_answer_states += "\nFiles: " + str(answer_state['files']) 
                formatted_answer_states += "\nAction: " + answer_state['action'] 
                # formatted_answer_states += "\nResult: " + answer_state['result'] 
                formatted_answer_states += "\nAnswer: " + answer_state['answer_state'] 
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most 5 of your most recent files, action, result, and answer, your goal is to choose and take the next best action and tool that you think could lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {self.available_actions.keys()}
            Most recent files, action, result, and answer states (oldest to newest):
            {formatted_answer_states}        
            """

            # 1) NEXT STEP AGENT: Ask for a direct next step for helping function calling API
            next_step = complete_text_openai(self.initial_prompt + "\nWhat is the next best action I should take. Be sure to look at the most recent action, result, and answer states because if I failed in completing a step, you should give me an easier next step. Only respond with the action I should take.", system_prompt=self.system_prompt, model=self.model)
            print("\nThis is the next step reported: ", next_step)

            # 2) FUNCTION CALLING AGENT: Call the function calling API by giving tools and available functions
            complete_text_openai(next_step, system_prompt=self.system_prompt, model=self.model, tools=self.tool_descriptions, available_functions=self.available_actions)

            # Add that information about the next step into the answer_state action column
            self.answer_states[-1]['action'] = "Assigned action: " + next_step + "\nTaken action: " + self.answer_states[-1]['action']

            completion = str(self.answer_states[-1])

            count += 1
            if count > MAX_STEPS:
                break

        return "Finished successfully! Final message: " + completion