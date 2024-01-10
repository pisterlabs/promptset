from dotenv import load_dotenv
load_dotenv()
import os
from langchain.llms  import OpenAI
from langchain.prompts import (
    PromptTemplate,
)
from voice.speech import speak
from voice.listen import listen

openai_api_key = os.getenv("OPENAI_API_KEY")


class Agent:
    def talk_to_god_with_template(self, text_from_user):
        template = """
You are omnipotent, kind, benevolent god. The user is "your child". Be a little bit condescending yet funny. You try to fulfill his every wish. Make witty comments about user wishes.

User: {input}
God: """
        llm = OpenAI(openai_api_key=openai_api_key)
        prompt = PromptTemplate.from_template(template)
        text = llm(prompt.format(input=text_from_user))
        speak(text)

    def run(self):
        listen(self.talk_to_god_with_template)


Agent().run()