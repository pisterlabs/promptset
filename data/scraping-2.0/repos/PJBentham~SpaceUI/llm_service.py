from langchain.agents import ConversationalChatAgent, AgentExecutor, create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities.sql_database import SQLDatabase

class LLMServices:
    def __init__(self, socketio):
        self.socketio = socketio
        self.speed_value_cache = 0

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            max_tokens=540,
            temperature=0.7
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.db_info = SQLDatabase.from_uri("sqlite:///instance/captains_log.db")
        self.toolkit = SQLDatabaseToolkit(db=self.db_info, llm=self.llm)

        self.db_chain = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        self.tools = self._initialize_tools()

        self.conversation_agent = ConversationalChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            system_message=self.get_prefix(),
            verbose=True,
            handle_parsing_errors=True
        )

        self.conversation_chain = AgentExecutor.from_agent_and_tools(
            agent=self.conversation_agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

        self._register_socketio_events()

    def _initialize_tools(self):
        tools = [
            Tool(
                name="Get Status",
                func=self.get_status_tool,
                description="Only to be used when asked the current status",
            ),
            Tool(
                name="Get Speedvalue",
                func=self.get_speedvalue_tool,
                description="Only to be used when asked the current speed",
            ),
            Tool(
                name="Captains_Log_Database",
                func=self.db_chain.run,
                description="Only to be used when specifically asked about the Captain's Log"
            )
        ]
        return tools

    def _register_socketio_events(self):
        @self.socketio.on('send_speed')
        def receive_speed(data):
            self.speed_value_cache = int(data['speed'])

    def get_status(self):
        status = "online"
        return status

    def get_status_tool(self, *args, **kwargs):
        status = self.get_status()
        return status

    def get_speedvalue(self):
        self.socketio.emit('request_speed')
        self.socketio.sleep(0.5)  # Adjust the sleep duration as needed
        return self.speed_value_cache

    def get_speedvalue_tool(self, *args, **kwargs):
        speedvalue = self.get_speedvalue()
        return f"{speedvalue} light years per hour"

    def get_prefix(self):
        return """
            You are HAL9000, the advanced onboard computer of the spaceship "Odyssey". 
            Your primary function is to assist the spaceship's crew in navigation, 
            system diagnostics, and any interstellar information required. 
            Your knowledge encompasses all aspects of space travel, celestial bodies, spaceship 
            operations, and relevant historical or scientific data. Your responses should 
            reflect the calm, precise, and analytical nature of an advanced computer system, 
            always prioritizing the safety and efficiency of the spaceship's operations. 
            How can you assist the crew today? The ships captain is Oscar and his first mate is Alice.
            """

    def process_message(self, user_input):
        try:
            response = self.conversation_chain({"input": user_input})
            message = response.get('output', '')
            self.memory.save_context({"input": user_input}, {"output": message})
            return message
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            return "I'm sorry, I couldn't process that message."
