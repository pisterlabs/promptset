import unittest

from langchain import PromptTemplate
from langchain.llms import FakeListLLM
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import Tool

from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.tools.utils import format_tools, format_tool_names, FinalAnswerTool


class TestSimpleAgent(unittest.TestCase):
    def test_run(self):
        answers = [
            '```json\n{\n\t"action": "check_weather",\n\t"action_input": "Moscow"\n',
            '```json\n{\n\t"action": "final_answer",\n\t"action_input": "In Moscow rainy with a temperature of 10°C."\n'
        ]
        llm = FakeListLLM(responses=answers)
        weather_tool = Tool(
            name="check_weather", description="Use it to check weather at some location",
            func=lambda x: "Rain, 10 C"
        )
        final_answer_tool = FinalAnswerTool()
        tools = [weather_tool, final_answer_tool]
        output_parser = ActionParser.from_extra_thoughts([], [])
        template = ChatPromptTemplate.from_messages(
            messages=[
                HumanMessagePromptTemplate.from_template("{{input}}", "jinja2"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                SystemMessage(content=format_tools(tools)),
                SystemMessage(content=PromptTemplate.from_template(
                    output_parser.get_format_instructions(), template_format="jinja2").format(
                    tool_names=format_tool_names(tools)
                ))
            ]
        )
        agent_executor = SimpleAgent.from_llm_and_prompt(
            llm=llm, prompt=template, output_parser=output_parser, stop_sequences=output_parser.stop_sequences
        ).get_executor(tools=tools)
        answer = agent_executor.run(input="What is the weather in Moscow?")
        self.assertEqual(answer, "In Moscow rainy with a temperature of 10°C.")
