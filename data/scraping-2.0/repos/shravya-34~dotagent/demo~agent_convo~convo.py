import os 
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from dotagent import compiler
from dotagent.agent.base_agent import BaseAgent
from dotagent.llms._openai import OpenAI
from dotagent.memory import SimpleMemory

path = Path(__file__).parent / 'conversation_prompt.hbs'
interview_template = Path(path).read_text()
interview_memory = SimpleMemory()

class Debate(BaseAgent):
    def __init__(self, 
                use_tools: bool = False,
                prompt_template: str = interview_template,
                memory = interview_memory,
                role = 'Republican',
                user_role = "Democrat",
                **kwargs):
        super().__init__(**kwargs)

        self.prompt_template = prompt_template
        self.use_tools = use_tools
        self.memory = memory
        self.llm = OpenAI(os.environ.get('OPENAI_MODEL'))
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        self.role = role
        self.user_role = user_role

        self.compiler = compiler(
            llm = self.llm,
            OPENAI_API_KEY = self.OPENAI_API_KEY,
            template = self.prompt_template,
            caching=kwargs.get('caching'),
            memory = self.memory,
            role = self.role,
            user_role = self.user_role 
        )

def main():
    role1_convo = " Online learning offers flexibility and convenience.It thus allows students to access educational resources from anywhere, at any time. This is particularly beneficial for non-traditional students, working adults, or those with busy schedules. "
    role1 = "Online Learning"
    role2 = "Traditional classroom learning"
    role1_agent = Debate(role=role1, user_role=role2)
    role2_agent = Debate(role=role2, user_role=role1)

    print(role1,": ",role1_convo)

    for _ in range(4):
        role2_convo = role2_agent.run(user_text=role1_convo)
        print(role2,": ",role2_convo)
        role1_convo = role1_agent.run(user_text=role2_convo)
        print(role1,": ",role1_convo)
    print("End of debate")

if __name__ == "__main__":
    main()