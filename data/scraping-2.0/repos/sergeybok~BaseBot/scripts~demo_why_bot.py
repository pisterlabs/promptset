from basebot import BaseBotWithLocalDb, BaseBot
from basebot import TheMessage, MessageWrapper
import openai

class WhyBot(BaseBotWithLocalDb):

    def help(self) -> str:
        return "I just respond back to your messages and follow it with Why? You need to modify me to make me do something interesting."

    def respond(self, message: MessageWrapper) -> MessageWrapper:
        if message.get_text():
            # context_messages = self.get_message_context(message, limit=5, descending=False) 
            response_text = message.get_text() +'? Why?'
            resp_message = self.get_message_to(user_id=message.get_sender_id())
            resp_message.set_text(response_text)
            return resp_message
        return {}

# initialize the bot, or bots
bot = WhyBot()

# Start the bot 
app = BaseBot.start_app(bot)

# you can provide as many bots as you'd like as arguments
#  to this function as long as they are all different classes
# example:
# app = BaseBot.start_app(bot, other_bot, other_other_bot)

