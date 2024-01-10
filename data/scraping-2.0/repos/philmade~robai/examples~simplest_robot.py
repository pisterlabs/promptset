from robai.memory import SimpleChatMemory
from robai.languagemodels import OpenAIChatCompletion
from robai.base import AIRobot
from robai.in_out import ChatMessage


class SimpleRobot(AIRobot):
    def __init__(self):
        super().__init__()
        # Look at SimpleChatMemory and you'll see that input_model is ChatMessage.
        # This robot therefore expectes a ChatMessage in process()
        self.memory = SimpleChatMemory(
            purpose="You're a robot that responds to a ChatMessage."
        )
        self.pre_call_chain = [
            self.remember_user_interaction,
            self.create_instructions,
        ]
        self.post_call_chain = [self.stop_the_robot]
        self.ai_model = OpenAIChatCompletion()

    def remember_user_interaction(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    def create_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [memory.system_prompt, memory.input_model]
        return memory

    """
    ROBOT NOW CALLS AI_MODEL USING MEMORY.INSTRUCTIONS_FOR_AI AS THE PROMPT, effectively doing this:
    robot.ai_model.call(memory.instructions_for_ai)
    """

    # ==== THE POST-CALL FUNCTIONS =====
    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # IF we're done, we have to call this or we will loop back to pre-call functions.
        memory.set_complete()
        return memory


if __name__ == "__main__":
    poetry_robot = SimpleRobot()
    some_input = ChatMessage(
        role="user",
        content="I'm a user input. Please write a limerick about an AI framework called Robai [Robe-Aye]",
    )
    memory = poetry_robot.process(some_input)
    our_result = memory.ai_response
    poetry_robot.console.pprint_message(our_result)
