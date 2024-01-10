from utils.localfile_loader import read_file
from .intent_executor import IntentExecutor
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents.tools import Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper

duckduckgo = DuckDuckGoSearchAPIWrapper(region='kr-kr')

class WebsearchIntentExecutor(IntentExecutor):
  def __init__(self, llm):
    tools =[
        Tool(
            name="search",
            func=duckduckgo.run,
            description="인터넷에 검색을 할 수 있습니다",
        )
    ]
    self.agent = initialize_agent(tools, llm, agent="zero-shot-react-description", handle_parsing_errors=True)
    self.translator = LLMChain(
      llm=llm,
      prompt=ChatPromptTemplate.from_template(
        template=read_file("./infrastructure/model/templates/translator_template.txt"),
      ),
      output_key="output",
    )

  def support(self, intent):
    return intent == "websearch"

  def execute(self, context):
    message = ChatPromptTemplate.from_template(
        template=read_file("./infrastructure/model/templates/websearch_template.txt"),
    ).invoke(context)
    print("[SYSTEM] Websearch 진행중...")
    result = self.agent.run(message)
    return self.translator.run(dict(request_message=result))

# def truncate_text(text, max_length=3000):
#   if len(text) > max_length:
#     truncated_text = text[:max_length - 3] + '...'
#   else:
#     truncated_text = text
#   return truncated_text


# def search(message):
#   return truncate_text(duckduckgo.run(message))