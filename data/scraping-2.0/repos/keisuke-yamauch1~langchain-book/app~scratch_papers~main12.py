import langchain
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

langchain.verbose = True
langchain.debug = True


def get_chat():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


if __name__ == "__main__":
    chat = get_chat()
    tools = load_tools(["terminal"])
    agent_chain = initialize_agent(
        tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    result = agent_chain.run("sample_data ディレクトリにあるファイルの一覧を教えて")
    print(result)



# {
#     "prompts": [
#         "Human: Answer the following questions as best you can. You have access to the following tools:
# terminal: Run shell commands on this Linux machine.
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [terminal]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
#
# Begin!
#
# Question: sample_data ディレクトリにあるファイルの一覧を教えて
# Thought:I need to list the files in the sample_data directory.
# Action: terminal
# Action Input: ls sample_data
# Observation: ls: cannot access 'sample_data': No such file or directory
#
# Thought:The sample_data directory does not exist. I need to find the correct directory.
# Action: terminal
# Action Input: ls
# Observation: Dockerfile
# app
# docker-compose.yml
# langchain
# requirements.txt
# venv
#
# Thought:The sample_data directory is not in the current directory. I need to search for it.
# Action: terminal
# Action Input: find / -name sample_data
# Observation: /usr/src/app/sample_data
# /opt/project/app/sample_data
#
# Thought:"
#     ]
# }