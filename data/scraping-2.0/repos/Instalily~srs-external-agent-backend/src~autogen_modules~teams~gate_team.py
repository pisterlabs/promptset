import os
import autogen
import guidance
from src.autogen_modules.helpers import base_config_gpt4
from typing import Optional, List, Dict, Any

import dotenv
dotenv.load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


GUIDANCE_SCRUM_MASTER_NLQ_PROMPT = """
Is the following block of text a marketing/ad related query? Please rank from 1 to 5, where:
1: Definitely not NLQ
2: Likely not NLQ
3: Neutral / Unsure
4: Likely NLQ
5: Definitely NLQ

Return the rank as a number exclusively using the rank variable to be casted as an integer.

Block of Text: {{potential_nlq}}
{{#select "rank" logprobs='logprobs'}} 1{{or}} 2{{or}} 3{{or}} 4{{or}} 5{{/select}}
"""

def gate_team():
    
    USER_PROXY_PROMPT = "A human admin."
    
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message=USER_PROXY_PROMPT,
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    scrum_agent = DefensiveScrumMasterAgent(
        name="Scrum_Master",
        llm_config=base_config_gpt4,
        system_message=GUIDANCE_SCRUM_MASTER_NLQ_PROMPT,
        human_input_mode="NEVER",
    )

    return [user_proxy, scrum_agent]

class DefensiveScrumMasterAgent(autogen.ConversableAgent):
    """
    Custom agent that uses the guidance function to determine if a message is a SQL NLQ
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the new reply function for this specific agent
        self.register_reply(self, self.check_nlq, position=0)

    def check_nlq(
        self,
        messages: Optional[List[Dict[any, any]]] = None,
        sender: Optional[autogen.Agent] = None,
        config: Optional[Any] = None,  # Persistent state.
    ):
        # Check the last received message
        last_message = messages[-1]["content"]

        # Use the guidance string to determine if the message is a SQL NLQ
        response = guidance(
            GUIDANCE_SCRUM_MASTER_NLQ_PROMPT, potential_nlq=last_message
        )

        # You can return the exact response or just a simplified version,
        # here we are just returning the rank for simplicity
        rank = response.get("choices", [{}])[0].get("rank", "3")

        return True, rank