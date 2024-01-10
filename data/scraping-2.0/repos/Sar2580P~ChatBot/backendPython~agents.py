from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from tools import *
from langchain.memory import ConversationSummaryBufferMemory


class PersonalAgent:
    def __init__(self, prev_summary):

        self.prev_summary = prev_summary
        self.prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
        self.suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        self.single_task_prompt = ZeroShotAgent.create_prompt(
            single_task_tools,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            # format_instructions = 
        )
        # memory
        self.memory = ConversationSummaryBufferMemory(llm = llm ,memory_key="chat_history", 
                                                      moving_summary_buffer = prev_summary)

        self.llm_chain = LLMChain(llm=llm, prompt=self.single_task_prompt)
        
        # agent
        self.agent = ZeroShotAgent(llm_chain=self.llm_chain, tools=single_task_tools, verbose=True)
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=single_task_tools, 
            verbose=True, memory=self.memory, handle_parsing_errors=True,
        )

    def run(self, query):
         try:
            return self.agent_chain.run(query)
         except Exception as e:
            print('I did not get that. Please try again.')
            return "I did not get that. Please try again."

    def get_chat_summary(self):
        messages = self.memory.chat_memory.messages
        self.prev_summary = (self.prev_summary +
                                self.memory.predict_new_summary(messages, self.prev_summary))
        
        return self.prev_summary
        
    
# print(agent_chain.run(input="How many people live in canada?"))
# print(agent_chain.run(input="who was the first prime minister of canada?"))
# agent_chain = PersonalAgent(prev_summary='My name is Sarvagya and I am a student at IIT Roorkee.')
# print(agent_chain.run("Explain what is in the coal mines act 1952?"))
# print(agent_chain.run("Explain in detail the provision in coal mines act 1952?"))
# print(agent_chain.run("What is my college name?"))
# print(agent_chain.run("What is it famous for?"))
# print(agent_chain.run("What is the name of the first prime minister of india?"))
# print(agent_chain.get_chat_summary())

