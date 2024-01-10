from langchain import OpenAI, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ConversationSummaryMemory, \
    ConversationSummaryBufferMemory, CombinedMemory

from susumu_ai_dialogue_system.infrastructure.config import Config, LangChainMemoryType


class LangChainPartsFactory:
    @staticmethod
    def create_summary_prompt() -> PromptTemplate:
        return PromptTemplate(
            input_variables=["summary", "new_lines"],
            template="""会話の内容を逐次まとめ、前回の要約に追加して新しい要約を作成します。
        例
        現在の要約:
        人間はAIが人工知能についてどう考えているか尋ねます。AIは人工知能は善意の力だと考えています。

        新しい会話の内容:
        人間: なぜ人工知能が善意の力だと考えているのですか？
        AI: 人工知能は人間がその可能性を最大限に引き出すのを助けるからです。

        新しい要約:
        人間はAIが人工知能についてどう考えているか尋ねます。AIは人工知能が善意の力であり、人間がその可能性を最大限に引き出すのを助けると考えています。
        例の終わり

        現在の要約:
        {summary}

        新しい会話の内容:
        {new_lines}

        新しい要約:""",
        )

    @staticmethod
    def create_memory(config: Config):
        memory_type = config.get_langchain_memory_type()
        match memory_type:
            case LangChainMemoryType.ConversationBufferMemory:
                return ConversationBufferMemory(return_messages=True)
            case LangChainMemoryType.ConversationBufferWindowMemory:
                return ConversationBufferWindowMemory(k=10, return_messages=True)
            case LangChainMemoryType.ConversationSummaryMemory:
                return ConversationSummaryMemory(
                    llm=OpenAI(),
                    input_key="input",
                    return_messages=True,
                    # prompt=LangChainPartsFactory.create_summary_prompt(),
                )
            case LangChainMemoryType.ConversationSummaryBufferMemory:
                return ConversationSummaryBufferMemory(
                    llm=OpenAI(),
                    max_token_limit=40,
                    return_messages=True,
                    # prompt=LangChainPartsFactory.create_summary_prompt(),
                )
            case LangChainMemoryType.CombinedMemory:
                conv_memory = ConversationBufferWindowMemory(k=10, return_messages=True)
                summary_memory = ConversationSummaryMemory(
                    llm=OpenAI(),
                    input_key="input",
                    return_messages=True,
                    # prompt=LangChainPartsFactory.create_summary_prompt(),
                )
                return CombinedMemory(memories=[conv_memory, summary_memory])
            case _:
                raise ValueError(f"Invalid memory_type: {memory_type}")
