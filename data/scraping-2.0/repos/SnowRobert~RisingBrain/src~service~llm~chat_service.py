"""Agnet Service"""
import time

from openai.error import RateLimitError

from src.common.utils import AGENT_NAME, GPT_MODEL
from rising_plugin.risingplugin import handle_chat_completion
from src.logs import logger
from src.model.chat_response_model import ChatResponseModel
from src.model.message_model import MessageModel


class ChatService:
    def __init__(self, ai_name=AGENT_NAME, llm_model=GPT_MODEL):
        self.ai_name = ai_name
        self.llm_model = llm_model

    def generate_context(self, prompt, relevant_memory, full_message_history, model):
        current_context = [
            # MessageModel.create_chat_message(
            #     "system", f"The current time and date is {time.strftime('%c')}"
            # ),
        ]

        # Add messages from the full message history until we reach the token limit
        next_message_to_add_index = len(full_message_history) - 1
        insertion_index = len(current_context)
        return (
            next_message_to_add_index,
            insertion_index,
            current_context,
        )

    # TODO: Change debug from hardcode to argument
    def chat_with_ai(
        self,
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
    ) -> ChatResponseModel:
        """Interact with the OpenAI API, sending the prompt, user input, message history,
        and permanent memory."""
        while True:
            try:
                """
                Interact with the OpenAI API, sending the prompt, user input,
                    message history, and permanent memory.

                Args:
                    prompt (str): The prompt explaining the rules to the AI.
                    user_input (str): The input from the user.
                    full_message_history (list): The list of all messages sent between the
                        user and the AI.
                    permanent_memory (Obj): The memory object containing the permanent
                      memory.
                    token_limit (int): The maximum number of tokens allowed in the API call.

                Returns:
                str: The AI's response.
                """
                model = self.llm_model  # TODO: Change model from hardcode to argument
                logger.debug(f"Chat with AI on model : {model}")

                # if len(full_message_history) == 0:
                #     relevant_memory = ""
                # else:
                #     recent_history = full_message_history[-5:]
                #     shuffle(recent_history)
                #     relevant_memories = permanent_memory.get_relevant(
                #         str(recent_history), 5
                #     )
                #     if relevant_memories:
                #         shuffle(relevant_memories)
                #     relevant_memory = str(relevant_memories)
                relevant_memory = ""
                # logger.debug(f"Memory Stats: {permanent_memory.get_stats()}")

                (
                    next_message_to_add_index,
                    insertion_index,
                    current_context,
                ) = self.generate_context(
                    prompt, relevant_memory, full_message_history, model
                )

                # while current_tokens_used > 2500:
                #     # remove memories until we are under 2500 tokens
                #     relevant_memory = relevant_memory[:-1]
                #     (
                #         next_message_to_add_index,
                #         current_tokens_used,
                #         insertion_index,
                #         current_context,
                #     ) = generate_context(
                #         prompt, relevant_memory, full_message_history, model
                #     )

                # Add Messages until the token limit is reached or there are no more messages to add.
                while next_message_to_add_index >= 0:
                    # print (f"CURRENT TOKENS USED: {current_tokens_used}")
                    message_to_add = full_message_history[next_message_to_add_index]

                    # Add the most recent message to the start of the current context,
                    #  after the two system prompts.
                    current_context.insert(insertion_index, message_to_add.to_json())

                    # Move to the next most recent message in the full message history
                    next_message_to_add_index -= 1

                # Append user input, the length of this is accounted for above
                current_context.extend(
                    [MessageModel.create_chat_message("user", user_input)]
                )

                logger.debug("------------ CONTEXT SENT TO AI ---------------")
                for message in current_context:
                    # Skip printing the prompt
                    if message["role"] == "system" and message["content"] == prompt:
                        continue
                    logger.debug(
                        f"{message['role'].capitalize()}: {message['content']}"
                    )
                    logger.debug("")
                logger.debug("----------- END OF CONTEXT ----------------")

                # TODO: use a model defined elsewhere, so that model can contain
                # temperature and other settings we care about
                return ChatResponseModel(
                    handle_chat_completion(model=model, messages=current_context)
                )
            except Exception as e:
                # TODO: When we switch to langchain, this is built in
                logger.warn("Error: ", "API Rate Limit Reached. Waiting 10 seconds...")
                raise e
