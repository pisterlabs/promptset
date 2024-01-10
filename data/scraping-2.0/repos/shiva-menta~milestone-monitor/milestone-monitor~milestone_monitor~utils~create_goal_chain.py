# Utils for making/getting the "create goal" LLM chain

from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory

from utils.goal_prompts import create_goal_chain_prompt
from utils.llm import BASE_LLM, BASE_CHATBOT_LLM


# Initializes a new create goal chain with new memory
def init_create_goal_chain(DEBUG=False) -> LLMChain:
    goal_create_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        ai_prefix="AI",
        human_prefix="User",
        input_key="input",
        return_messages=True,
        k=3,
    )

    create_goal_chain = LLMChain(
        llm=BASE_CHATBOT_LLM,
        prompt=create_goal_chain_prompt,
        verbose=True,
        memory=goal_create_memory,
    )

    return create_goal_chain, goal_create_memory


# Gets a create goal chain with existing memory
def get_create_goal_chain(memory: ConversationBufferWindowMemory, DEBUG=False) -> int:
    create_goal_chain = LLMChain(
        llm=BASE_CHATBOT_LLM,
        prompt=create_goal_chain_prompt,
        verbose=True,
        memory=memory,
    )
    print(memory)

    return create_goal_chain
