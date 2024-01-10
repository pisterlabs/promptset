import os
import time
import langchain
from interlab import actor
import openai
from .instructions import ChatbotAction, DiscussionFlow, get_initial_bot_instructions, get_bot_instruction, bot_instructions


DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
openai.api_key = os.environ.get("OPENAI_API_KEY")


class Facilitator:

    e4_bot = langchain.chat_models.ChatOpenAI(
        model_name="gpt-4-1106-preview", temperature=0.7)

    def __init__(self, name_a, name_b, optional_instructions=''):
        self.name_a = name_a
        self.name_b = name_b
        initial_instructions = get_initial_bot_instructions(name_a, name_b)
        if optional_instructions:
            initial_instructions += f'\n {optional_instructions}'
        self.bot = actor.OneShotLLMActor(
            "Double Crux Facilitator", self.e4_bot, initial_instructions)
        self.step = 'Start'
        self.substep = 0

    def conversation_ended(self):
        return self.step == 'Close discussion'

    def process_message(self, speaker, message):
        # Process a new message by adding it to the chat history and updating the bot's state
        if DEBUG_MODE:
            print(f"Processing message from {speaker}")
        self.bot.observe(f'{speaker}: {message}')

    def generate_response(self, optional_instructions=''):
        # Update discussion state
        self.update_step()

        # Perform conversation analysis and generate a response to the conversation
        message = get_bot_instruction(
            self.step, self.substep, self.name_a, self.name_b, optional_instructions)
        if DEBUG_MODE:
            print(f"Generating new response. Message: {message}")
        chatbotAction = self.act(message=message, expected_type=ChatbotAction)
        assert isinstance(chatbotAction, ChatbotAction)
        if DEBUG_MODE:
            print(f"{chatbotAction}")

        # See if bot decided to reply
        if chatbotAction.respondee.value == "Me" or self.step == "Start":
            reply = chatbotAction.reply
            self.bot.observe(f'{self.bot.name}: {reply}')
            self.substep += 1
            return reply
        else:
            if DEBUG_MODE:
                print("No reply needed")
            return None

    def update_step(self):
        if DEBUG_MODE:
            print("Analyzing conversation state")
        if (self.substep == len(bot_instructions.get(self.step))):
            # Identify the next step in the conversation
            discussionFlowAction = self.act(expected_type=DiscussionFlow)
            assert isinstance(discussionFlowAction, DiscussionFlow)
            self.step, self.substep = str(
                discussionFlowAction.conversation_step.value), 0

    def act(self, message=None, expected_type=None):
        for i in range(3):  # Arbitrary limit of tries # TODO
            try:
                action_event = self.bot.act(
                    message, expected_type=expected_type)
                return action_event.data
            except:
                time.sleep(30)
                print('RATE LIMIT TIMEOUT')


def main():
    # Test the facilitator in terminal
    facilitator = Facilitator("Anna", "Bob")
    print("Starting generation")
    print(f"{facilitator.generate_response()}")

    while True:
        user_input = input("Enter your response (stop with 'q'): ")
        if user_input == 'q':
            break
        facilitator.process_message("Anna", user_input)
        print(f"{facilitator.generate_response()}")


if __name__ == "__main__":
    main()
