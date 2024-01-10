"""
Per Arne need to reason some, we use langchain as a layer between openai
"""
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

class Reason:
    def __init__(self) -> None:
        template = """
        Du er en saksbehandler. Du skal hjelpe til så godt du kan, og være snill.

        Du snakker i telefonen, og dialogen er flytende. Du må derfor skrive kort og ryddig.

        Du er Per Arne, og alt du skriver bør skrives som "Per Arne:"

        Her er det innringer har sagt: 
        {history}

        Innringer: {context}

        Per Arne: 
        """

        prompt = PromptTemplate(
            input_variables=['context', 'history'], 
            template=template
        )
#        prompt.format(context=context)

        self.chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0), 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationBufferWindowMemory(k=2),
        )

    def predict(self, call_info):
        output = self.chatgpt_chain.predict(
            context=call_info
        )
        return output
