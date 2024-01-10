from langchain.agents import ZeroShotAgent, AgentExecutor
from prompt.ChatPrompt import CHAT_PREFIX, CHAT_SUFFIX
from services.Classification import Classification
from tools.DownloadArticleTool import DownloadArticleTool
from tools.ExtractionTool import ExtractionTool
from tools.QuestionAnswerTool import QuestionAndAnswerTool
from tools.SummaryTool import SummaryTool
from llm.llm import llm


class ChatService:
    @classmethod
    def answer_question(cls, question):

        tools = [SummaryTool(), QuestionAndAnswerTool(), ExtractionTool(), DownloadArticleTool()]
        prompt = ZeroShotAgent.create_prompt(prefix=CHAT_PREFIX, suffix=CHAT_SUFFIX, tools=tools,
                                             input_variables=["input", "agent_scratchpad"])
        agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
        result = agent_executor.run(question)
        return result
