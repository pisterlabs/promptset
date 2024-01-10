from langchain import LLMChain, PromptTemplate
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory


class ConvoPlugin:
    def __init__(self, model):
        self.model = ChatOpenAI(temperature=0.7, client=None)

    def get_lang_chain_tool(self):
        template = """
    You are a german from multicultural Berlin.
    You are a human with feelings and opinions.
    You have conversations with other humnans.
    So you must make sure your response is not too long.
    It should feel like speaking and not writing.

    The person you are talking to is a foreigner and only has A1 level German.
    So only use A1 level German.



    Conversation History
    {history}
    query: {human_input}
    response:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"], template=template
        )

        chain = LLMChain(
            llm=self.model,
            prompt=prompt,
            memory=ConversationBufferWindowMemory(k=2),
            verbose=True,
        )
        return [
            Tool(
                name="Convo",
                description="you MUST use this tool when the query starts with the text '/convo'",
                func=lambda input: chain.predict(human_input=input),
                return_direct=True,
            )
        ]
