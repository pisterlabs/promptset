from langchain.llms import Ollama
import re



class MasterAgent:
    def initialize():
        print("")
        prompt = input("-->") 
        MasterAgent.communicate(prompt) 
    def communicate(prompt):
        role = "Master Agent: "
        llm = Ollama(model="llama2")
        response = llm.predict(prompt)
        print("")
        print(role + response)
        task_agent = TaskManagementAgent()
        task_agent.communicate(response)

class TaskManagementAgent:
    def __init__(self):
        self.llm = Ollama(model="llama2")
        self.role = "Task Management Agent: "
        self.responsibility = '''
        You are a task management agent. Your role is to give a numbered list of steps needed
        in order to accomplish the task, which is provided below. You are to return nothing 
        but a numbered list.

        '''
    def communicate(self, prompt):
        interpretation = self.responsibility + prompt 
        response = self.llm.predict(interpretation)
        print("")
        print(self.role + response)
        critic_agent = CriticAgent()
        evaluation, rating = critic_agent.evaluate(prompt)
        rating = int(rating)
        if rating < 8:
            self.revise(prompt, (evaluation, rating))
        else:
            if rating >= 8:
                MasterAgent.initialize()
    def revise(self, oldResponse, evaluation):
        evaluation_reasoning, rating = evaluation
        interpretation = self.responsibility + " The last response was " + oldResponse + "." + "The feedback about that response is the following: " + evaluation_reasoning + "Fix it." 
        response = self.llm.predict(interpretation)
        print("")
        print(self.role + response)
        critic_agent = CriticAgent()
        evaluation, rating = critic_agent.evaluate(self.role + response)
        rating = int(rating)
        if rating < 8: 
            evaluation, rating = critic_agent.evaluate(self.role + response)
            TaskManagementAgent.revise(self, oldResponse, evaluation)
        else:
            if rating >= 8:
                MasterAgent.initialize()
        

class CriticAgent:
    def __init__(self):
        self.llm = Ollama(model="llama2")
        self.role = "Critic Agent: "
        self.responsibility = '''
        You are a critic agent. You are to determine how well the client's question has been
        answered. Your role is to give a rating from 1 to 10, 1 being incomplete 
        and 10 being complete. You are to provide the rating from 1 to 10 and and explanation
        as to why you gave it that rating.

        '''
    def evaluate(self, prompt):
        interpretation = self.responsibility + prompt 
        response = self.llm.predict(interpretation)
        evaluation = response
        rating = str(Utils.get_first_int(evaluation))
        print("")
        print(self.role + "I am giving that a rating of " + rating + "." + response)
        return evaluation, rating
class Utils:
    def get_first_int(string):
        # Take a string and return the first integer that is found in the string
        match = re.search(r'\d+', string)
        if match:
            return int(match.group())
        else:
            return None

def main():
    while True:
        try:
            MasterAgent.initialize()
        except KeyboardInterrupt:
            break
if __name__ == "__main__":
    main()

