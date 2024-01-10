"""
written by:   Eugene M.
              https://github.com/apexDev37

date:         dec-2023

demo:         Reviving-Memories: query an LLM based on previous conversations.
"""


from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from utils.helpers import (
    init_interactive_conversation,
    read_conversation_from,
    setup_for_query,
    setup_gpt_model,
    update_memory_context,
)


def main() -> None:
    """
    Script entry-point func.
    """

    llm = setup_gpt_model(version="3.5", variant="turbo")
    memory = ConversationBufferMemory()

    # Create a chain to store history of conversations
    # using the `BufferMemory` and pass as context to the LLM.
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False,
    )

    # Load past conversation and pass to the LLM's memory context.
    conversations = read_conversation_from(file="memory/data/conversation.txt")
    update_memory_context(memory, conversations)

    setup_for_query(memory)

    # Launch an interactive conversation in your terminal.
    for user_prompt in init_interactive_conversation():
        ai_response = conversation.predict(input=user_prompt)
        print("[AI] >>>", ai_response)


if __name__ == "__main__":
    main()
