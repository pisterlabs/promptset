import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_hub.tools.bing_search import BingSearchToolSpec
from llama_index.agent import OpenAIAgent


class WebChatbot:
    def __init__(self,  engine="gpt-3.5-turbo", system_prompt="Answer all questions without bias in one sentence"):
        load_dotenv()
        self.key = os.getenv("BING_SUBSCRIPTION_KEY")
        self.tool_spec = BingSearchToolSpec(api_key=self.key)
        self.agent = OpenAIAgent.from_tools(self.tool_spec.to_tool_list())
        self.engine = engine 
        self.client = OpenAI()
        self.system_prompt = system_prompt

    def search_web(self, query):
        val = self.agent.chat(query)
        return val

    def set_prompt_with_context(self, context):
        return f"For the text in between <<>> \
        respond using the context in () \
        <<{self.system_prompt}>> ({context}) :"

    def get_chat_completion(self,prompt):        
        response = self.client.chat.completions.create(
        model=self.engine,
        messages=[
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": prompt}
        ]
        )
        return response

    def chat_loop(self):
        while True:
            user_input = input("You: ")
            result  = self.search_web(user_input)
            search_result = self.set_prompt_with_context(result)
            if user_input.lower() == "exit":
                print("Chatbot: Goodbye!")
                break
            if search_result:
                result = self.get_chat_completion(str(search_result))

                print("Chatbot: ",result.choices[0].message.content)

chat = WebChatbot()
chat.chat_loop()