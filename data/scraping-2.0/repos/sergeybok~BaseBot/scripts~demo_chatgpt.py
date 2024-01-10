from basebot import BaseBotWithLocalDb, BaseBot
from basebot import TheMessage, MessageWrapper
import openai

class ChatGPTBot(BaseBotWithLocalDb):
    def help(self) -> str:
        return "I am a wrapper around ChatGPT. Ask me anything and I will do my best to respond."
    def respond(self, message: MessageWrapper) -> MessageWrapper:
        if message.get_text():
            # get previous messages, oldest message first
            context_messages = self.get_message_context(message, limit=5, descending=False) 
            chatgpt_messages = []
            for msg in context_messages:
                if msg.get_sender_id() == message.get_sender_id() and msg.get_text():
                    chatgpt_messages.append({'role': 'user', 'content': msg.get_text()})
                elif msg.get_text():
                    chatgpt_messages.append({'role': 'assistant', 'content': msg.get_text()})
            # add current message last
            chatgpt_messages.append({'role': 'user', 'content': message.get_text()})
            # Call OpenAI API (this will fail without API key)
            chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=chatgpt_messages)
            response_text = chatgpt_response['choices'][0]['message']['content']
            resp_message = self.get_message_to(user_id=message.get_sender_id())
            resp_message.set_text(response_text)
            return resp_message
        return {}

# initialize the bot, or bots
bot = ChatGPTBot()

# Start the bot 
app = BaseBot.start_app(bot)

# you can provide as many bots as you'd like as arguments
#  to this function as long as they are all different classes
# example:
# app = BaseBot.start_app(bot, other_bot, other_other_bot)

