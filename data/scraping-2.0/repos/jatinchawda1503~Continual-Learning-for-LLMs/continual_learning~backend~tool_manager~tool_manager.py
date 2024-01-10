from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain


class MemoryAgentTool:
    def __init__(self,llm, memory, prompt_handler):
        self.memory_prompt_handler = prompt_handler
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
