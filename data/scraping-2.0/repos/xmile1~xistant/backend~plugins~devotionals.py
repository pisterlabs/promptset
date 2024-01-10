from itertools import chain
from typing import Any
from langchain.agents import load_tools
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from config import settings
from langchain import LLMChain, PromptTemplate

class DevotionalsPlugin():
  def __init__(self, model):
      self.model = ChatOpenAI(temperature=1.0, max_tokens=1024, client=None)
      self.theologian_prompt = PromptTemplate.from_template(
          """You are a Christian and theologian
          Create a devotional message for the chapter after {previous_chapter}, it should be in an interesting story telling fashion but must try to be as factual as possible.

          In a another section, add a Reflection and Application section, where you reflect on the chapter and apply it practically to today's world.

          In another section, act as a seasoned archeologist and critic and add an unbiased proof of the facts and refute the information that is not historical fact or unverified.

          In another section, add a useful life hack.

          The first line of your response should be the book and chapter of the bible that you are writing about e.g. Isaiah 1
          and the next line should be the devotional message

          Token limit: 1024
          """
      )
      self.chain = LLMChain(llm=self.model, prompt=self.theologian_prompt, verbose=True)
      
  def get_lang_chain_tool(self):
    return [DevotionalsPluginTool(chain=self.chain)]

class DevotionalsPluginTool(BaseTool):
    name = "Daily devotionals generator"
    description = "This tool generates a daily devotional, input should be a chapter of the bible"
    return_direct = True
    chain: LLMChain

    def _run(self, query: str) -> str:
        previous_chapter_location = "data/devotionals/previous_chapter.txt"

        f = open(previous_chapter_location, "r")
        previous_chapter = f.read()

        response = self.chain.run(previous_chapter=previous_chapter)

        f = open(previous_chapter_location, "w")
        f.write(response.split('\n')[0])

        return f"Good morning, here is a devotional excerpt for you. {response}"
    
    async def _arun(self, query: str) -> str:
        """Use the Devotional tool asynchronously."""
        raise NotImplementedError("Daily devotional does not support async")
