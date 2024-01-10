from langchain.schema import BaseMemory

from eurelis_kb_framework.base_factory import ParamsDictFactory


class ConversationBufferMemoryFactory(ParamsDictFactory[BaseMemory]):
    OPTIONAL_PARAMS = {
        "ai_prefix",
        "human_prefix",
        "input_key",
        "output_key",
        "memory_key",
        "return_messages",
    }

    def __init__(self):
        super().__init__()
        self.params.update(
            {
                "memory_key": "chat_history",
                "return_messages": True,
                "input_key": "question",
                "output_key": "answer",
            }
        )

    def build(self, context: "BaseContext") -> BaseMemory:
        params = self.get_optional_params()

        from langchain.memory import ConversationBufferMemory

        return ConversationBufferMemory(**params)
