from lmtk.bots import BaseBot, register_bot

@register_bot('langchain')
class LangChainBot(BaseBot):

  title = 'LangChain'
  loader_delay = 0

  def respond(self, query):
    yield self.chain.predict(input=query)

  def load(self, state):
    # Imports are here to keep the langchain dependency optional
    from langchain.llms import OpenAI
    from langchain import ConversationChain

    temperature = self.profile.config.get('temperature', .3)
    llm = OpenAI(temperature=temperature)

    self.chain = ConversationChain(llm=llm)
    self.chain.memory.buffer = state.get('buffer', '')

  def save(self):
    return { 'buffer': self.chain.memory.buffer }

  def inspect(self):
    return self.chain.prompt.format(
      input='[ next input ]',
      history=self.chain.memory.buffer
    )

  def get_buffer(self, name):
    return ('history', self.chain.memory.buffer, '.txt')

  def set_buffer(self, name, value):
    self.chain.memory.buffer = value
