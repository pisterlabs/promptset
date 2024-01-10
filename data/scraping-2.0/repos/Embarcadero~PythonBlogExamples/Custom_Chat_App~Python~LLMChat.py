from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


class LLM():
    def __init__(self, name, age) -> None:
        self.age = age
        self.name = name

        llm = OpenAI(openai_api_key="your_openai_api_key",
                     temperature=0)  # Create instance of openai llm
        temp = "You are a nice chatbot having a conversation with a human of age {}.".format(
            age)
        template = temp + """ So respond appropriately and preferably in around 2 sentences.

        Previous conversation:
        {chat_history}

        New human question: {question}
        Response:"""
        prompt = PromptTemplate.from_template(
            template)  # Create template for prompt to openai
        # Create buffer memory for multiturn conversation
        memory = ConversationBufferMemory(memory_key="chat_history")
        self.conversation = LLMChain(  # Chain all components
            llm=llm,
            prompt=prompt,
            memory=memory
        )

    def getReply(self, qs):
        # Provide conversation to LLM to get appropriate answer
        response = self.conversation({"question": qs})
        return response["text"]
