import openai
import datetime

class GPT_Agent:
    def initialize(self, name):
        raise NotImplementedError("Subclasses must implement the 'initialize' method.")
    
    def _make_goals(self, user_msg):
        raise NotImplementedError("Subclasses must implement the '_make_goals' method.")
    
    def make_first_decision(self, user_msg):
        raise NotImplementedError("Subclasses must implement the 'make_first_decision' method.")
    
    def make_decision(self,  relevant_memory, last_user_msg, assistant_msg, command_result, user_msg):
        raise NotImplementedError("Subclasses must implement the 'make_decision' method.")

    def make_summary(self, user_msg: str, text_to_summarize: str, base_summary: str = None) -> tuple[str, str]:
        raise NotImplementedError("Subclasses must implement the 'make_decision' method.")
    
    def print_response(self, response):
        raise NotImplementedError("Subclasses must implement the 'response' method.")

