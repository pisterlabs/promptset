from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


class Conversation:
    def __init__(self, llm, messages, ai_prefix="", human_prefix=""):
        """
        Initializes a conversation chain with an AI language model using Langchain.

        Parameters:
            llm: The language model instance (e.g., LLama).
            messages: A list of dictionaries containing role and content of each message.
            ai_prefix: Prefix to prepend to assistant messages (optional).
            human_prefix: Prefix to prepend to user messages (optional).

        Returns:
            None
        """
        self.user_input = None
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.messages = messages
        self.memory_buffer = ConversationBufferWindowMemory(
            k=6, return_messages=True, ai_prefix=ai_prefix, human_prefix=human_prefix
        )
        prompt_template = self.process_content()
        self.conversation = ConversationChain(
            llm=llm,
            verbose=False,
            memory=self.memory_buffer
        )
        self.conversation.prompt.template = prompt_template

    def process_content(self):
        """
        Processes the list of messages, updating the chat memory and extracting a prompt template.

        This method iterates through a list of messages and performs the following actions:
        1. For each "system" message, appends its content to the prompt template.
        2. For each "user" message, adds the user's message to the langchain chat memory.
        3. For each "assistant" message, adds the AI's response to the langchain chat memory.

        Parameters:
            self: The class instance (assuming this method is part of a class).

        Returns:
            None

        Note:
            - The prompt template will be used to build a coherent context for the AI's responses.
            - The chat memory keeps track of the conversation history for better contextual responses.
        """
        messages = self.messages

        prompt_template = ""
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                prompt_template += message["content"] + "\n\n"
                postfix_prompt = "\n\nCurrent conversation:\n{history}\n{Human}: {input}\n{AI}:"
                prompt_template += postfix_prompt
                prompt_template = prompt_template.replace(
                    "{Human}", self.human_prefix).replace(
                    "{AI}", self.ai_prefix)
            if message["role"] == "user":
                if idx == len(messages) - 1:
                    self.user_input = message["content"]
                else:
                    self.memory_buffer.chat_memory.add_user_message(message["content"])
                    self.user_input = "continue"

            if message["role"] == "assistant":
                self.memory_buffer.chat_memory.add_ai_message(message["content"])

        return prompt_template

    def predict(self):
        response = self.conversation.predict(input=self.user_input)

        # This deletes the stop sequence triggered word.
        stop_word = f"\n{self.human_prefix}"
        if response.endswith(stop_word):
            filtered_response = response[:-len(stop_word)]
        else:
            filtered_response = response

        return filtered_response
