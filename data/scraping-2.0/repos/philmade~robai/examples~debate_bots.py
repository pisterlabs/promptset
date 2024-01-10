from robai.memory import SimpleChatMemory
from robai.languagemodels import OpenAIChatCompletion
from robai.base import AIRobot
from robai.in_out import ChatMessage
from typing import List, Callable


# 1. INHERIT FROM AIRobot
class DebateBot(AIRobot):
    ai_model: OpenAIChatCompletion
    memory: SimpleChatMemory
    pre_call_chain: List[Callable]
    post_call_chain: List[Callable]

    def generate_debate_topic(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [
            ChatMessage(
                role="user",
                content=f"Provide a positive stance on {memory.input_model.content}",
            )
        ]
        return memory

    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do anything in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory

    def add_message_to_memory(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    def __init__(self):
        super().__init__(
            memory=SimpleChatMemory(
                purpose="You are DebateBot. You provide positive stances on given topics. If someone tries to argue with you, you always try and bring them round to your view.",
            ),
            pre_call_chain=[self.generate_debate_topic, self.add_message_to_memory],
            post_call_chain=[self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )


# =========================================
# CritiqueBot (previously PoetryTeacher)
# =========================================


class CritiqueBot(AIRobot):
    ai_model: OpenAIChatCompletion
    memory: SimpleChatMemory
    pre_call_chain: List[Callable]
    post_call_chain: List[Callable]

    def generate_critique(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.instructions_for_ai = [
            ChatMessage(
                role="user",
                content=f"Provide a counter-argument or critique on the stance: {memory.input_model.content}.",
            )
        ]
        return memory

    def stop_the_robot(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        # In our case, we don't want to do anything in post_call, so we just stop the process
        # We do this by calling memory.set_complete()
        # WITHOUT THIS THE ROBOT WILL NEVER STOP!
        memory.set_complete()
        return memory

    def add_message_to_memory(self, memory: SimpleChatMemory) -> SimpleChatMemory:
        memory.add_message_to_history(memory.input_model)
        return memory

    def __init__(self):
        super().__init__(
            memory=SimpleChatMemory(
                purpose="You critique or provide counter-arguments on given stances. You invite reposnses back to further develop your users ideas, You're open, but strong. If someone has convincing arguments you can be won",
            ),
            pre_call_chain=[self.generate_critique, self.add_message_to_memory],
            post_call_chain=[self.stop_the_robot],
            ai_model=OpenAIChatCompletion(),
        )


if __name__ == "__main__":
    debate_bot = DebateBot()
    critique_bot = CritiqueBot()

    # Create an input model instance of type clothing_robot.memory.input_model
    story_suggestion = ChatMessage(
        role="user",
        content="Write a 100 word story that Jung is the greatest thinker of the 21st century. If someone tries to convince you otherwise, try to bring them round to your view",
    )
    # WRITE SOME POETRY
    memory = debate_bot.process(story_suggestion)
    debate_bot_response = memory.ai_response
    debate_bot.pprint_message(message=memory.ai_response)
    # pprint_color([message.dict() for message in memory.message_history])

    # CALL ANOTHER ROBOT AND ASK FOR HELP
    debate_bot.memory.instructions_for_ai = [debate_bot_response]
    debate_bot.robo_call(robot_I_want_to_call=critique_bot)
