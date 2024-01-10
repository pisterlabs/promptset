from .configs import *
from langchain.agents import AgentType, create_sql_agent
from langchain.chat_models import AzureChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
class Assistant:
    def __init__(self, prefix) -> None:
        self.prefix = prefix
        self.llm = AzureChatOpenAI(
            deployment_name=DEPLOYMENT_NAME,
            openai_api_base=OPENAI_API_BASE,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=OPENAI_API_KEY,
            openai_api_type=OPENAI_API_TYPE,
            model_name="gpt-4", 
        )
        HOST = os.getenv('HOST')
        NAME = os.getenv('NAME')
        DBUSER = os.getenv('DBUSER')
        PORT = os.getenv('PORT')
        PASSWORD = os.getenv('PASSWORD')
        CONNECTION_URL = f'postgresql://{DBUSER}:{PASSWORD}@{HOST}:{PORT}/{NAME}'
        # CONNECTION_URL = 'sqlite:///C:/Users/Minh Long/Documents/IUH/Semester 8/CNM/Vis4Teacher_CK/Vis4T_main/db.sqlite3'
        engine = create_engine(CONNECTION_URL)
        db = SQLDatabase(engine, include_tables=[
            "Teacher", "Subject_class", "Subject", "Subject_student", "Student", "University_class"
        ])
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            # verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
            
    def run(self, question):
        return self.agent_executor.run(self.prefix + question)
