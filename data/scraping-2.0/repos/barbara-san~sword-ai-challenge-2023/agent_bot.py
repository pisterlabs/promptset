from langchain.chat_models import AzureChatOpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

import json
from langchain.schema import messages_from_dict, messages_to_dict, AIMessage

from subjects_and_tools import *

def get_context(subject):
    return f"""\
    The following text is simply an guide that the AI must follow after receiving this text input. \
    You won't need to use tools to answer to this prompt, as it only exists to give context to the AI. \
    For any following prompt given you can use tools to achieve an answer.
    This is a friendly conversation between a human and an AI. \
    The AI serves the purpose of being a {subject} tutor. \
    The human is considered to be a highschooler learning in the Portuguese schools, which follow the portuguese teaching system. \
    The AI should help the human by giving clear detail on how to solve an exercise or how to understand a concept. \
    The AI must not use techniques that are considered to be of higher educational experience. \
    If the AI does not know the answer to a question, it truthfully says it does not know. \
    You have access to the following tools:\
    """

def get_prompt(subject):
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    return ZeroShotAgent.create_prompt(
        TOOLS_OF[subject],
        prefix=get_context(subject),
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )


class AgentChatBot:
    def __init__(self, subject) -> None:
        self.SUBJECT = subject
        self.MAX_TOKENS = 1000
        self.TEMPERATURE = 0.2
        self.VERBOSE = True
        self.LLM = AzureChatOpenAI(deployment_name="gpt35-team-3-0301", max_tokens=self.MAX_TOKENS, temperature=self.TEMPERATURE)
        
        # create empty memory and save an empty json file for the memory
        self.MEMORY = ConversationBufferMemory(memory_key="chat_history")
        self.MEMORY_PATH = f"json_chats_memory/{self.SUBJECT}.json"
        self.save_memory() # to create the history file

        self.LLM_CHAIN = LLMChain(llm=self.LLM, prompt=get_prompt(subject))
        self.AGENT = ZeroShotAgent(llm_chain=self.LLM_CHAIN, tools=TOOLS_OF[subject], verbose=self.VERBOSE)
        self.AGENT_CHAIN = AgentExecutor.from_agent_and_tools(
            agent=self.AGENT, tools=TOOLS_OF[subject], verbose=True, memory=self.MEMORY
        )

    def load_memory(self):
        with open(self.MEMORY_PATH, "r") as history:
            self.MEMORY.chat_memory.messages = messages_from_dict(json.load(history))
        history.close()

    def save_memory(self):
        with open(self.MEMORY_PATH, "w") as history:
            json.dump(messages_to_dict(self.MEMORY.chat_memory.messages), history)
        history.close()
    
    def ask(self, input_prompt):
        self.load_memory()
        try:
            answer = self.AGENT_CHAIN.run(input_prompt)
            self.save_memory()
            return answer
            
        except Exception as e:
            answer = str(e)
            if answer.startswith("Could not parse LLM output: `"):
                answer = answer.removeprefix("Could not parse LLM output: `").removesuffix("`")
                self.MEMORY.chat_memory.messages.append(
                    AIMessage(content=answer)
                )
                self.save_memory()
                return answer
            else:
                raise Exception(str(e))
    