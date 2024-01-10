from agent.main_agent import create_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict
from data import get_chat_memory, check_user_exists
from langchain.memory import ChatMessageHistory

agents_by_user = {}


def agent_for_user(user_token, final_output_handler=None):
    chat_agent = agents_by_user.get(user_token)

    if chat_agent is None:
        memory = create_memoy(user_token)
        chat_agent = create_agent(memory, final_output_handler)
        agents_by_user[user_token] = chat_agent

    return chat_agent


def create_memoy(user_token):
    """Loads chat history from database into a memory object for an agent"""
    # Only to get the user id
    user_id = check_user_exists(user_token)
    messages = get_chat_memory(user_id)

    # Convert messages to memory format
    memory_messages = [
        {
            "type": "human" if msg["message_type"] == "human" else "ai",
            "data": {"content": msg["content"], "additional_kwargs": {}},
        }
        for msg in messages
    ]

    history = ChatMessageHistory()
    history.messages = messages_from_dict(memory_messages)

    memory = ConversationBufferMemory(
        memory_key="memory",
        return_messages=True,
        chat_memory=history,
    )

    return memory


if __name__ == "__main__":
    agent = agent_for_user("1234567892")
    print(agent("What was the last question I asked you?"))
