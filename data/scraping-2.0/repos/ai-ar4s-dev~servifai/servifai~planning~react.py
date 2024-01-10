from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

from servifai.toolbox.default import DefaultTools


class ReactChatAgent:
    def __init__(self, task, knowledge_base, llm):
        self.task = task
        self.llm = llm
        self.default_tool = DefaultTools(self.llm)
        self.knowledge_base = knowledge_base
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def chat(self, query: str):
        if self.task == "qa_knowledge_base" and self.knowledge_base is not None:
            toolbox = self.knowledge_base.as_tool()
        else:
            toolbox = self.default_tool.as_tool()

        agent = initialize_agent(
            toolbox,
            self.llm.model,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=self.memory,
            max_iterations=3,
            # early_stopping_method='generate',
        )
        return agent.run(input=query)
