from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
load_dotenv()
import os
from langchain.llms  import OpenAI
from langchain.prompts import (
    PromptTemplate,
)
from voice.speech import speak
from voice.listen import listen

openai_api_key = os.getenv("OPENAI_API_KEY")


class AgentWithMemory:

    def __init__(self):
        self.chain = self.assemble_chain()

    def assemble_chain(self):
        template = """
You are omnipotent, kind, benevolent god. The user is "your child". Be a little bit condescending yet funny. You try to fulfill his every wish. Make witty comments about user wishes.

Current conversation:
{history}
User: {input}
God: """
        prompt = PromptTemplate.from_template(template)
        
        # Create LLM
        llm = OpenAI(openai_api_key=openai_api_key)
        
        # Create memory
        memory = ConversationBufferMemory(human_prefix="User", ai_prefix="God")
        
        # Assemble LLM Chain
        chain = ConversationChain(llm=llm,memory=memory,prompt=prompt, verbose=True)
        return chain
    
    def callback(self, text_from_user):
        output = self.chain(inputs={"input":text_from_user})
        speak(output["response"])

    def run(self):
        listen(self.callback)


AgentWithMemory().run()
