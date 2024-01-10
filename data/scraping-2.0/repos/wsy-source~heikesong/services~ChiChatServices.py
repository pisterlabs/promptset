from langchain import LLMChain, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from llm.llm import llm
from prompt.ChatPrompt import CHAT_PROMPT


class ChiChat:
    @classmethod
    def chat(cls, question):
        template = PromptTemplate.from_template(CHAT_PROMPT)
        # system_message = SystemMessage(content=CHAT_PROMPT)
        # human_message = HumanMessage(content=question)
        # llm([system_message, human_message])
        chain = LLMChain(llm=llm, verbose=True, prompt=template)
        answer = chain.run(question)
        return answer
