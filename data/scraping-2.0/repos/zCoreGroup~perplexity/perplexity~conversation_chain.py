import asyncio
from typing import AsyncGenerator
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from perplexity.models.model_classes import MODEL_CLASSES
from .templates import CHAT_PROMPT_TEMPLATE


class StreamingConversationChain:
    """
    Class for handling streaming conversation chains.
    It creates and stores memory for each conversation,
    and generates responses using the provided language model.
    """

    def __init__(self, model_type: str, api_key: str, temperature: float = 0.0):
        self.memories = {}
        self.api_key = api_key
        self.temperature = temperature
        self.model_type = model_type

    def get_model(self):
        callback_handler = AsyncIteratorCallbackHandler()
        model_info = MODEL_CLASSES.get(self.model_type)

        if model_info is None:
            raise ValueError(f"Invalid model type: {self.model_type}")

        ModelClass = model_info["class"]
        api_key_name = model_info["api_key_name"]

        return ModelClass(
            callbacks=[callback_handler],
            streaming=True,
            temperature=self.temperature,
            **{api_key_name: self.api_key},
        )

    async def generate_response(
        self, conversation_id: str, message: str
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous function to generate a response for a conversation.
        It creates a new conversation chain for each message and uses a
        callback handler to stream responses as they're generated.
        :param conversation_id: The ID of the conversation.
        :param message: The message from the user.
        """
        language_model = self.get_model()

        memory = self.memories.get(conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self.memories[conversation_id] = memory

        chain = ConversationChain(
            memory=memory,
            prompt=CHAT_PROMPT_TEMPLATE,
            llm=language_model,
        )

        run = asyncio.create_task(chain.arun(input=message))

        async for token in language_model.callbacks[0].aiter():
            yield token

        await run


# Now you can create an instance of StreamingConversationChain with either language model
# scc_openai = StreamingConversationChain('ChatOpenAI', 'YOUR_OPENAI_API_KEY', 0.5)
# scc_anthropic = StreamingConversationChain('ChatAnthropic', 'YOUR_ANTHROPIC_API_KEY', 0.5)
