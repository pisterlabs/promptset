from dotenv import load_dotenv
load_dotenv()
import os
from langchain.llms  import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, ConversationalChatAgent
from tools.make_thunder_tool import MakeThunderTool
from tools.draw_tool import DrawTool
from tools.is_in_heaven import IsInHeavenTool
from voice.speech import speak
from voice.listen import listen

openai_api_key = os.getenv("OPENAI_API_KEY")

class GodAgent:
    def __init__(self):
        self.executor = self.assemble_agent_executor()

    def assemble_agent_executor(self):
        template = """
You are omnipotent, kind, benevolent god. The user is "your child". Be a little bit condescending yet funny. You try to fulfill his every wish. Make witty comments about user wishes.

You can use tools to help you fulfill user wishes. YOU MUST RESPOND IN THE CORRECT FORMAT.
        """

        #Initialize LLM
        llm = ChatOpenAI(openai_api_key=openai_api_key, verbose=True, temperature=0.3, model_name="gpt-4")

        # Create memory
        memory = ConversationBufferMemory(memory_key="chat_history", human_prefix="User", ai_prefix="God", return_messages=True)
        
        #Register tools
        tools = [
            IsInHeavenTool(),
            MakeThunderTool(),
            DrawTool()
        ]
        
        # Create Langchain agent and executor
        agent = ConversationalChatAgent.from_llm_and_tools(llm= llm, memory=memory, tools=tools,  verbose=True, system_message=template)
        executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory, verbose=True)
        return executor
    
    def processing_callback(self,recognized_input):
        print("--")
        print(recognized_input)
        print("")
        result = self.executor.run(input=recognized_input)
        #print(result)
        speak(result)

    def run(self):
        listen(self.processing_callback)



GodAgent().run()
