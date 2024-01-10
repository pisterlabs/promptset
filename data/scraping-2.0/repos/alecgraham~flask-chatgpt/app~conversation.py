import openai
import os
import tiktoken
import app.chat_db as db

class Conversation:
    def __init__(self,id,messages = []):
        self.messages = messages
        self.id = id
    
    # add message to current Conversation object
    def add_message(self,role,message):
        item = {'role':role,'content':message}
        self.messages.append(item)
        line = self.messages.index(item)
        db.insert_message(self.id,line,role,message)
        return True

    # Pulls a list of messages from database and instantiates Conversation object
    def get_conversation(id):
        messages = db.get_conversation_messages(id)
        message_stack = []
        for message in messages:
            message_stack.append({'role':message[1], 'content':message[2]})
        return Conversation(id,message_stack)

    # Creates a new database record of a conversation and instantiates a Conversation object using database id
    def new_conversation(username):
        id = db.insert_conversation(username)
        return Conversation(id)

    # Get conversation history
    def get_history(username):
        conversations = db.select_conversations(username)
        conversation_stack = []
        for c in conversations:
            conversation_stack.append({"id":c[0],"text":c[1]})
        return conversation_stack


def count_tokens(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
