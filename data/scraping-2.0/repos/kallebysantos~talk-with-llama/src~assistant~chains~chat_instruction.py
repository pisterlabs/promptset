from langchain import LlamaCpp, LLMChain, PromptTemplate

from .._assistant import Assistant

class ChatInstruction(Assistant):
    """
    Chat data structure.
    """
    def __init__(self, model_path: str):
        self.model = LlamaCpp(
            verbose=True,
            model_path=model_path, 
            callbacks=[self.handler],
            temperature=0.4,
            n_ctx=1024,
            max_tokens=2048,
            last_n_tokens_size = 16,
            repeat_penalty=1.1,
            stop=["User:"]
        )
            
    def new_chain(self):
        return LLMChain(
            llm=self.model,
            prompt=self.get_prompt_template(), 
            callbacks=[self.handler]
        )

    def get_prompt_template(self):
        template = """
This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.
Continue the chat dialogue below. Write a single reply for the character "Assistant".

User: What's the capital of France?\n\n
Assistant: Paris is the city you're looking for.\n\n
User: What means "Bom dia" ?\n\n
Assistant: Good morning in Portuguese.\n\n
User: {question}\n\n
Assistant:"""

        return PromptTemplate(
            template=template, 
            input_variables=["question"], 
        )