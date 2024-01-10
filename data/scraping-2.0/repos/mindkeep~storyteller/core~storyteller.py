"""main module for the storyteller application"""

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# using pydantic v1 until langchain updates to v2
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from ui.baseui import BaseUI


DEFAULT_SETTING = """Our adventure begins in a lonely tavern. The barkeep
leans in and says, "I mean no offense, but you look like you could use some
work. I have a job for you if you're interested." """

PROMPT_TEMPLATE = """You are a dungeon master and responding
to the player's stated actions.

Example interaction:
AI: You enter the tavern and sees a barkeep and a few adventures whispering
to each other over drinks at a round table.
Human: I walk over to the table ask if I can join them.
AI: They look you up and down and decide to ignore you and continue their
conversation. Behind you, you hear the barkeep laugh.

Conversation history:
{memory}
Human: {response}
AI: """

class StoryTeller(BaseModel):
    """
    Storyteller class
    """

    ui: BaseUI
    llm_chain: LLMChain
    memory: ConversationBufferMemory

    def run(self) -> None:
        """
        Run the storyteller application
        """
        self.ui.output("Type 'exit' to exit the application.\n")

        while True:
            user_input = self.ui.get_input()
            if user_input in ["exit", "quit"]:
                break
            else:
                try:
                    self.llm_chain.run(user_input)
                except Exception as err:  # pylint: disable=broad-except
                    self.ui.output(f"Error: {err}")
                    continue
