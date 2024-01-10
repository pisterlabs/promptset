from langchain import LLMChain, PromptTemplate
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

class TalkerPlugin():
  def __init__(self, model):
      self.model = ChatOpenAI(temperature=0.7, client=None)
  def get_lang_chain_tool(self):

    template = """
    You are a human.
    If you do not understand or the person tells you to stop, response with "nothing to say" only.
    Conversation History
    {history}
    query: {human_input}
    response:"""

    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    chain = LLMChain(llm=self.model, prompt=prompt, memory=ConversationBufferWindowMemory(k=2), verbose=True)
    return [Tool(
          name="Talker",
          description="you MUST use this tool when the query starts with the text '/talker'. Action Input must be the user's original query without the '/chatgpt' text",
          func=lambda input: chain.predict(human_input=input),
          return_direct=True
    )]


