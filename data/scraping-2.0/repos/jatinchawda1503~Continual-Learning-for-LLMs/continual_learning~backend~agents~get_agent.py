from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType, Tool

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain


class MemoryAgentTool:
    def __init__(self,llm, memory, memory_prompt_handler):
        self.memory_prompt_handler = memory_prompt_handler
        self.llm = llm
        self.memory = memory
        
    def handle_duplicate_memory(self):
        conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            prompt=PromptTemplate(input_variables=["history", "input"], template=self.memory_prompt_handler.DUPLICATE_MEMORY_HANDLE_PROMPT)
        )
        return conversation
    
    def handle_temporal_memory(self):
        conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            prompt=PromptTemplate(input_variables=["history", "input"], template=self.memory_prompt_handler.TEMPORARY_MEMORY_HANDLE_PROMPT)
        )
        return conversation
    
    def handle_episodic_memory(self):
        conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            prompt=PromptTemplate(input_variables=["history", "input"], template=self.memory_prompt_handler.EPISODIC_MEMORY_HANDLE_PROMPT)
        )
        return conversation
    
    def handle_contradictory_memory(self):
        conversation = ConversationChain(
            llm=self.llm, 
            memory=self.memory,
            prompt=PromptTemplate(input_variables=["history", "input"], template=self.memory_prompt_handler.CONTRADICTORY_MEMORY_HANDLE_PROMPT)

        )
        return conversation


class MemoryConversationalAgent:
    def __init__(self,llms,history,memory_prompt_handler, template):
        self.history = history
        self.memory  = ConversationSummaryBufferMemory(chat_memory=self.history, llm=llms,k=6, return_messages=True,verbose=True)
        self.llm = llms
        self.memory_agent_tool = MemoryAgentTool(llm=self.llm, memory=self.memory, memory_prompt_handler=memory_prompt_handler)
        self.conversation_template = template
        self.tools = self.init_tools()
    def init_tools(self):
        tool = [
            Tool(
                name="Duplicate Memory Tool",
                description="good in handling conversation with duplicate memory",
                func=self.memory_agent_tool.handle_duplicate_memory().run,
            ),
            Tool(
                name="Contradictory Memory Tool",
                description="good in handling conversation with  contradictory memory",
                func=self.memory_agent_tool.handle_contradictory_memory().run,
            ),
            Tool(
                name="Temporal Memory Tool",
                description="good in handling conversation with  temporal memory",
                func=self.memory_agent_tool.handle_temporal_memory().run,
            ),
            Tool(
                name="Episodic Memory Tool",
                description="good in handling conversation with  episodic memory",
                func=self.memory_agent_tool.handle_episodic_memory().run,
            ),
        ]
        
        return tool
    
    def get_agent(self):
        agent = initialize_agent(
        agent_type = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        llm=self.llm,
        memory=self.memory,
        tools=self.tools,
        verbose=True,
        handle_parsing_errors=True,
        pronpt = PromptTemplate(input_variables=["history", "input", "agent_scratchpad"], template=self.conversation_template),
    )
        return agent