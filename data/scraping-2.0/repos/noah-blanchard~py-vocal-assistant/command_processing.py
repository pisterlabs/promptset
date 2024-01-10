from openai_agent import OpenAIAgent

class CommandProcessing:
    def __init__(self):
        self.openai_agent = OpenAIAgent()

    def handle_command(self, command):
        return self.openai_agent.get_command_label(command)
    
    def get_approve_deny(self, command):
        return self.openai_agent.get_approve_deny(command)