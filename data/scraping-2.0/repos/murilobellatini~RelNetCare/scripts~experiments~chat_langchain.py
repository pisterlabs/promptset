import click
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

class MyChatbot:
    def __init__(self):
        template = """
        Assistant is a large language model trained by OpenAI.


        {history}
        Human: {human_input}
        Assistant:"""

        self.prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

        self.chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0),
            prompt=self.prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=2),
        )

    def chat(self, input):
        return self.chatgpt_chain.predict(human_input=input)

@click.command()
@click.argument('input', type=str)
def cli(input):
    chatbot = MyChatbot()
    print(chatbot.chat(input))

if __name__ == '__main__':
    cli()
