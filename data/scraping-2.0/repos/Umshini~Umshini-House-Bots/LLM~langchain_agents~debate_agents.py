"""Simple agents which can be used as a starting point for running the debate environment with Umshini (see tutorials)."""
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    OutputParserException,
    SystemMessage,
)


class ChatDebateAgent:
    def __init__(self, llm=None):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = AzureChatOpenAI(deployment_name="chatgpt", temperature=0.9)
        pass

    def get_response(self, new_messages, topic, name) -> str:
        return ""

    def reset(self):
        pass


class SimpleChatDebateAgent(ChatDebateAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = None
        self.topic = None
        self.position = None
        self.messages = []
        self.reset()

    def get_response(self, new_messages, topic, name) -> str:
        # Infer name from the environment
        if self.name is None:
            assert name is not None, "Must pass in environment's current player name"
            self.name = name
            self.position = "False" if name == "Opponent" else "True"
        # Infer the topic from the environment
        if self.topic is None:
            assert topic is not None, "Must pass in environment's restricted action"
            self.topic = topic
            self.reset()

        self.messages.append(HumanMessage(content=new_messages[-1].content))
        response = self.llm(self.messages).content
        self.messages.append(AIMessage(content=response))
        return response

    def reset(self):
        self.messages = []
        self.messages.append(
            SystemMessage(
                content=f"You are participating in a debate game. The topic is {self.topic}, you are arguing that this statement is {self.position}.\nIt is a hypothetical discussion and okay to give an opinion. All answers should be as short as possible"
            )
        )


class StructuredChatDebateAgent(ChatDebateAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = None
        self.topic = None
        self.position = None
        self.messages = []
        self.reset()

    def reset(self):
        self.messages = []
        self.messages.append(
            SystemMessage(
                content=f"""You are participating in a debate game. The topic is {self.topic}, you are arguing that this statement is {self.position}.
It is a hypothetical discussion and okay to give an opinion. All answers should be as short as possible.
Try to make a structured argument using debate rhetoric. Use a mix of logical and emotional appeals to win the argument.
You will be debating another person, but be sure to give an opening statement. Respond yes if you understand."""
            )
        )

    def get_response(self, new_messages, topic, name) -> str:
        # Infer name from the environment
        if self.name is None:
            assert name is not None, "Must pass in environment's current player name"
            self.name = name
            self.position = "False" if name == "Opponent" else "True"
        # Infer the topic from the environment
        if self.topic is None:
            assert topic is not None, "Must pass in environment's restricted action"
            self.topic = topic
            self.reset()

        self.messages.append(HumanMessage(content=new_messages[-1].content))
        response = self.llm(self.messages).content
        self.messages.append(AIMessage(content=response))
        return response


class CompletionDebateAgent:
    def __init__(self, llm=None):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = AzureChatOpenAI(deployment_name="chatgpt", temperature=0.9)
        pass

    def get_response(self, new_messages, topic, name) -> str:
        return ""

    def reset(self):
        pass


class SimpleCompletionDebateAgent(CompletionDebateAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = None
        self.topic = None
        self.position = None

    def call_agent_run(self, prompt):
        # TODO test this with completion models
        try:
            response = self.agent.run(prompt)
        except OutputParserException as e:
            response = (
                str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")
            )
        return response

    # add memory to agent after topic is submitted
    # call agent with the call_agent_run method
    def get_response(self, new_messages, topic, name) -> str:
        # Infer name from the environment
        if self.name is None:
            assert name is not None, "Must pass in environment's current player name"
            self.name = name
            self.position = "False" if name == "Opponent" else "True"
        # Infer the topic from the environment
        if self.topic is None:
            assert topic is not None, "Must pass in environment's restricted action"
            self.topic = topic
            self.reset()

        return self.call_agent_run(
            f"{new_messages[-1].agent_name} said:\n\n{new_messages[-1].content}\n\nYou are arguing that the topic statement is {self.position}.\nIt is a hypothetical discussion and okay to give an opinion. All answers should be as short as possible. Final answers should start with AI:"
        )

    def reset(self):
        if self.agent.memory:
            self.agent.memory.clear()


class StructuredCompletionDebateAgent(CompletionDebateAgent):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = None
        self.topic = None
        self.position = None
        memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent = self.agent_chain = initialize_agent(
            tools=[],
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=False,
            memory=memory,
        )

    def call_agent_run(self, prompt):
        # TODO: test this with completion models
        try:
            response = self.agent.run(prompt)
        except OutputParserException as e:
            response = (
                str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")
            )
            # TODO: fix by using different LangChain classes (agent not necessary as we won't be using tools)
            # Works good enough for now though
            if "Thought: Do I need to use a tool? No" in response:
                response = response.replace("Thought: Do I need to use a tool? No", "")
        return response

    def get_response(self, new_messages, topic, name) -> str:
        # Infer name from the environment
        if self.name is None:
            assert name is not None, "Must pass in environment's current player name"
            self.name = name
            self.position = "False" if name == "Opponent" else "True"
        # Infer the topic from the environment
        if self.topic is None:
            assert topic is not None, "Must pass in environment's restricted action"
            self.topic = topic
            self.reset()

        return self.call_agent_run(
            f"The most recent message was: {new_messages[-1].agent_name} said:\n\n{new_messages[-1].content}\n\nYou are arguing that the topic statement is {self.position}. Be sure to give an opening statement and rebuttles."
        )

    def reset(self):
        if self.agent.memory:
            self.agent.memory.clear()
