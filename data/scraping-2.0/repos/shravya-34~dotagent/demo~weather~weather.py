import json
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

path = Path(__file__).parent / 'prompt.hbs'
interview_template = Path(path).read_text()
interview_memory = SimpleMemory()

functions = [ 
    {"name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string", 
                "enum": ["celsius", "fahrenheit"]
            },
        },
        "required": ["location"],
    }
    }
]

def get_current_weather(location, unit="fahrenheit"):
    weather_info = {
        "location": location,
        "temperature": "71",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


class Weather(BaseAgent):
    def __init__(self, 
                use_tools: bool = False,
                prompt_template: str = interview_template,
                memory = interview_memory,
                name = 'Weather',
                **kwargs):
        super().__init__(**kwargs)

        self.prompt_template = prompt_template
        self.use_tools = use_tools
        self.memory = memory
        self.name = name
        self.llm = OpenAI(os.environ.get('OPENAI_MODEL'))
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        self.functions = functions
        self.get_current_weather = get_current_weather

        self.compiler = compiler(
            llm = self.llm,
            OPENAI_API_KEY = self.OPENAI_API_KEY,
            template = self.prompt_template,
            caching=kwargs.get('caching'),
            memory = self.memory,
            name = self.name ,
            functions = self.functions,
            get_current_weather = get_current_weather
        )
        
agent = Weather()
print(agent.run(user_text = "hello!?"))
print(agent.run(user_text = "Hows the weather today in Mumbai ?"))
# print(agent.run(user_text = "Sure, I am a 4th year engineering student and I have been practicing code for a while now."))
# print(agent.run(user_text = 'Hows the weather today in mumbai ?'))