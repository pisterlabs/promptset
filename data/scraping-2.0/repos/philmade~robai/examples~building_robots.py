from robai.memory import SimpleChatMemory
from robai.languagemodels import OpenAIChatCompletion
from robai.base import AIRobot
from robai.in_out import ChatMessage
from typing import List, Callable


class YourPoetryRobot(AIRobot):
    # Start with the init and fill it in as you go.
    # You'll need: memory, a pre_call_chain, a post_call_chain and an ai_model
    # You need to call super().__init__() so the base AIRobot sets everything up properly.

    # ====== INIT YOUR ROBOT ========
    def __init__(self):
        super().__init__()
        self.memory = SimpleChatMemory(
            purpose="You're a robot that writes poems from the user's input."
        )
        self.pre_call_chain = [
            self.create_log,
            self.remember_user_interaction,
            self.create_instructions,
        ]
        self.post_call_chain = [self.log_what_I_did, self.stop_the_robot]
        self.ai_model = OpenAIChatCompletion()

    # ====== THE PRE-CALL FUNCTIONS  =====
    def create_log(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        self.console.pprint("Here's what was put into robot.process")
        self.console.pprint(message=memory.input_model)
        return memory

    def remember_user_interaction(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    def create_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        """
        The line below shows a very simple set of instructions.
        You MUST set memory.instructions_for_ai or the AI won't know what to do!
        The 'system_prompt' is available in memory and it will always be:
            ChatMessage(role='system', content='The Purpose of the Robot')
        It is also added to message_history on init, so you could also get it like this:
        system_prompt = memory.message_history[0]

        So here, our instructions for the AI are the system_prompt message, plus the input_model instance which
        is populated with whatever the user input. Input_models are always ChatMessage objects.
        Instructions_for_ai must always be a list of ChatMessage objects.
        """
        memory.instructions_for_ai = [memory.system_prompt, memory.input_model]

        # It's equivalent to this:
        memory.instructions_for_ai = [
            ChatMessage(role="system", content=self.memory.purpose),
            ChatMessage(
                role="user",
                content="Whatever the user said when the robot was called, like, 'Hey, I heard you're great at writing poems?!",
            ),
        ]
        # But let's set the instructions back to the dynamically created values.
        memory.instructions_for_ai = [memory.system_prompt, memory.input_model]
        return memory

    """
    ROBOT NOW CALLS AI_MODEL USING MEMORY.INSTRUCTIONS_FOR_AI AS THE PROMPT, effectively doing this:
    robot.ai_model.call(memory.instructions_for_ai)
    """

    # ==== THE POST-CALL FUNCTIONS =====
    def log_what_I_did(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        self.console.print("[white]My instructions were:")
        self.console.pprint(memory.instructions_for_ai)
        self.console.print("[white]The AI responded with:")
        self.console.pprint(memory.ai_response)
        self.console.print("[white]Here is the AI's full memory")
        self.console.pprint(memory)
        return memory

    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do much else in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory


if __name__ == "__main__":
    poetry_robot = YourPoetryRobot()
    some_input_might_be = ChatMessage(
        role="user",
        content="I'm a user input. I heard that no matter what I say, you'll write a poem about it?",
    )
    result = poetry_robot.process(some_input_might_be)
