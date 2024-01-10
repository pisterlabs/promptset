import openai


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"


class ChatAgent:

    """
    ChatAgent class for interacting with OpenAI Chatcomplete API
    
    Attributes:
        agent_role (str): Role of the chat agent (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM)
        conversation (list): List of messages in the conversation
        chat_model (str): OpenAI Chatcomplete API model ID    
    """

    def __init__(self, role=ROLE_ASSISTANT, system_msg = "You are a helpful AI assistant.", 
            api_model="local", context_size = 0):

        """
        The constructor for ChatAgent class.
        """

        self.agent_role = role
        self.conversation = []
        self.chat_model = api_model
        self.context_size = context_size
        self.tokens_used = 0

        starting_prompt = self.make_message(ROLE_SYSTEM, system_msg)
        self.conversation.append(starting_prompt)


    def make_message(self, role, message):

        """
        Create a message object for OpenAI Chatcomplete API
        
        Args:
            role (str): Role of the chat agent (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM)
            message (str): Message content
        
        Returns:
            dict: Message object
        """

        return {
            "role": role,
            "content": message
        }
    

    def add_to_conversation(self, message):

        """
        Add a message to the conversation and get a response from OpenAI Chatcomplete API
        
        Args:
            message (str): Message content
            
        Returns:
            str: Response from OpenAI Chatcomplete API
        """

        user_message = self.make_message(ROLE_USER, message)
        self.conversation.append(user_message)

        # Send conversation history to OpenAI Chatcomplete API and print response
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=self.conversation
        )
        
        response_content = response.choices[0].message.content.strip()
        bot_reply = self.make_message(ROLE_ASSISTANT, response_content)
        self.conversation.append(bot_reply)
        self.tokens_used += response.usage.total_tokens

        return response_content

