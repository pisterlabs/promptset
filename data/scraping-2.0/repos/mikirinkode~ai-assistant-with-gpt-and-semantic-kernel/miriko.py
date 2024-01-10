from openai import OpenAI
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

class Miriko:
    def __init__(self, openai_model, api_key, org_id):
        # initialize Miriko
        self.name = "Miriko"
        self.memories = [
            {"role": "system", "content": "You are Miriko, A personal AI Assistant that helps User daily tasks."}
        ]
        
        # initialize OpenAI client
        self.openai_client = OpenAI(api_key=api_key)
        self.openai_model = openai_model
        
        # Initialize kernel
        self.kernel = sk.Kernel()
        chat_service = OpenAIChatCompletion(openai_model, api_key, org_id)
        # Register chat service
        self.kernel.add_chat_service("OpenAI_chat_gpt", chat_service)
        
        # import created skill
        self.skill = self.kernel.import_semantic_skill_from_directory("./skills", "MirikoSkill")
        
        self.brainstormer = self.skill["ExpertBrainstorming"]
        self.summarizer = self.skill["Summarizer"]
        
    def use_skill(self, skill_name, prompt):
        if skill_name == "Expert Brainstorming":
            return self.brainstormer(prompt)
        elif skill_name == "Summarizer":
            return self.summarizer(prompt)
        
    
    def chat(self, prompt):
        # add prompt to memory so miriko can remember it
        self.memories.append({"role": "user", "content": prompt})
        
        result = self.openai_client.chat.completions.create(
            model = self.openai_model,
            messages= self.memories,
            stream=True,
        )
        # system_content = {"role": "system", "content": result.choices[0].delta.content}
        # self.memories.append(system_content)
        
        # return result
        return result
    
    def get_all_miriko_memory(self):
        return self.memories