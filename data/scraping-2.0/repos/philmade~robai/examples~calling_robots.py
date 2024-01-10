from robai.memory import SimpleChatMemory
from robai.languagemodels import OpenAIChatCompletion
from robai.base import AIRobot
from robai.in_out import ChatMessage
from typing import List, Callable


# =========================================
# HOW TO BUILD A ROBOT
# =========================================
# 1. INHERIT FROM AIRobot
class YourRobot(AIRobot):
    # ==========
    # 2. SET TYPE HINTS TO HELP YOU IMAGINE WHAT YOU'RE DOING
    # ALWAYS PUT THEM
    # ==========
    ai_model: OpenAIChatCompletion  # <--- This is a subclass of BaseAIModel. You can make your own very easily or get one from robai.languagemodels
    memory: SimpleChatMemory  # <--- Start with SimpleChatMemory, You can change this later
    pre_call_chain: List[Callable]  # <--- a list of functions
    post_call_chain: List[Callable]  # <--- a list of functions

    # =========================================
    # 3. DEFINE SOME PRE-CALL FUNCTIONS
    # =========================================
    def create_poem_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [memory.message_history[0], memory.input_model]

        return memory

    def add_message_to_memory(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    # =========================================
    # 4. DEFINE SOME POST-CALL FUNCTIONS
    # =========================================
    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do anything in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory

    # =========================================
    # 5. SET UP THE ROBOT AT INIT, TELLING IT EXACTLY WHAT TO USE
    # =========================================
    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
    def __init__(self):
        super().__init__(
            # These are placeholders for now, we'll develop these properly later
            memory=SimpleChatMemory(
                purpose="You are a short story robot. You write wonderful short stories for users. You can also accept help from other robots"
            ),
            pre_call_chain=[self.add_message_to_memory, self.create_poem_instructions],
            post_call_chain=[self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )


class StoryRobot(AIRobot):
    ai_model: OpenAIChatCompletion
    pre_call_chain: List[Callable]
    post_call_chain: List[Callable]
    memory: SimpleChatMemory

    # PRECALL METHODS
    def create_poem_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [memory.message_history[0], memory.input_model]

        return memory

    def add_message_to_memory(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    # POSTCALL METHODS
    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do anything in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # Without this the robot loops back to pre_call functions > call_ai > post_call
        # It will continue until its memory is set to complete somewhere in the chain!
        memory.set_complete()
        return memory

    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
    def __init__(self):
        super().__init__(
            memory=SimpleChatMemory(
                purpose="You are a short story robot. You write wonderful short stories for users. You can also accept help from other robots",
            ),
            pre_call_chain=[self.create_poem_instructions, self.add_message_to_memory],
            post_call_chain=[self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )


class PoetryTeacher(AIRobot):
    ai_model: OpenAIChatCompletion
    pre_call_chain: List[Callable]
    post_call_chain: List[Callable]
    memory: SimpleChatMemory

    # PRECALL METHODS
    def create_poem_instructions(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [memory.message_history[0], memory.input_model]
        self.ai_model.max_tokens = 40
        return memory

    def add_message_to_memory(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    # POSTCALL METHODS
    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do anything in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory

    # WHEN BUILDING A ROBOT WE HAVE TO CALL SUPER() AND PASS IN THE MEMORY CLASS
    def __init__(self):
        super().__init__(
            memory=SimpleChatMemory(
                purpose="You are a poetry teacher. You help people write better via the art of poetry. You can also accept help from other robots and give help to other robots",
            ),
            pre_call_chain=[self.create_poem_instructions, self.add_message_to_memory],
            post_call_chain=[self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )


if __name__ == "__main__":
    poetry_robot = StoryRobot()
    poetry_teacher = PoetryTeacher()

    # Create an input model instance of type clothing_robot.memory.input_model
    story_suggestion = ChatMessage(
        role="user", content="Write a 30 word story that would change the world"
    )
    # WRITE SOME POETRY
    memory = poetry_robot.process(story_suggestion)

    # CALL ANOTHER ROBOT AND ASK FOR HELP
    poetry_robot.robo_call(robot_I_want_to_call=poetry_teacher)
