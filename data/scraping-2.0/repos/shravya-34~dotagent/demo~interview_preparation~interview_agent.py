import os 
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(r'D:/DotagentDemo/dotagent')
from dotagent import compiler
from dotagent.agent.base_agent import BaseAgent
from dotagent.llms._openai import OpenAI
from dotagent.memory import SimpleMemory

path = Path(__file__).parent / 'template.hbs'
interview_template = Path(path).read_text()
interview_memory = SimpleMemory()

function = [
    {"name": "add_numbers",
     "description": "Adds two numbers",
     "parameters": {
        "a": "number",
        "b": "number"
      }
    }]

class Interview(BaseAgent):
    def __init__(self, 
                use_tools: bool = False,
                prompt_template: str = interview_template,
                memory = interview_memory,
                name = 'Interprep',
                **kwargs):
        super().__init__(**kwargs)

        self.prompt_template = prompt_template
        self.use_tools = use_tools
        self.memory = memory
        self.name = name
        self.llm = OpenAI(os.environ.get('OPENAI_MODEL'))
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        self.function = function

        self.compiler = compiler(
            llm = self.llm,
            OPENAI_API_KEY = self.OPENAI_API_KEY,
            template = self.prompt_template,
            caching=kwargs.get('caching'),
            memory = self.memory,
            name = self.name ,
            function = self.function
        )
        
agent = Interview()
print(agent.run(user_text = "hello!?"))
print(agent.run(user_text = "I want to prepare for software development role"))
print(agent.run(user_text = "Sure, I am a 4th year engineering student and I have been practicing code for a while now."))
print(agent.run(user_text = 'End the test.'))