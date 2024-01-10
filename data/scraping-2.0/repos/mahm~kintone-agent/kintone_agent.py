from typing import Dict, Any, Tuple, Generator
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from openai import BadRequestError
from pathlib import Path

from tools.kintone import get_apps, get_app, get_records, add_record
from tools.request import get_url, get_url_head
import prompt

MODEL_NAME = "gpt-4"


class KintoneAgent:
    def __init__(self, working_directory: Path) -> None:
        self.intermediate_steps = []
        self.working_directory = working_directory
        self.tools = self.setup_tools()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in self.tools]
        )

        self.agent_chain = self.setup_chain(llm_with_tools, is_fallback=False)
        self.fallback_chain = self.setup_chain(llm, is_fallback=True)

    def setup_tools(self) -> list:
        return [get_app, get_records, add_record, get_url, get_url_head]

    def setup_chain(self, llm: ChatOpenAI, is_fallback: bool) -> Any:
        prompt = self.create_prompt(is_fallback)
        assigns = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: self.memory.load_memory_variables({})[
                "chat_history"
            ],
        }
        if is_fallback:
            return assigns | prompt | llm | StrOutputParser()
        else:
            return assigns | prompt | llm | OpenAIFunctionsAgentOutputParser()

    def create_prompt(self, is_fallback: bool) -> ChatPromptTemplate:
        system_message = prompt.FALLBACK_PROMPT if is_fallback else prompt.AGENT_PROMPT
        kintone_information = self.generate_kintone_information_prompt()
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", kintone_information),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def generate_kintone_information_prompt(self) -> str:
        result = get_apps.run({})
        apps = []
        for app in result["apps"]:
            apps.append(
                f"app_id: {app['appId']}, name: {app['name']}, description: {app['description']}"
            )
        joined_apps = "\n".join(apps)
        return f"The list of kintone applications you have access to is as follows You should not attempt to access any other applications other than these: \n{joined_apps}"

    async def run(self, input_message: str) -> Generator[Tuple[str, bool], None, None]:
        while True:
            message, is_final = await self.process_step(input_message)
            yield {"message": message, "is_final": is_final}
            if is_final:
                self.finish_process(input_message, message)
                break

    async def process_step(self, input_message: str) -> Tuple[str, bool]:
        try:
            output = await self.agent_chain.ainvoke(
                {"input": input_message, "intermediate_steps": self.intermediate_steps}
            )
            if isinstance(output, AgentFinish):
                return output.return_values["output"], True
            else:
                observation = self.tool_execute(output.tool, output.tool_input)
                message = self.format_tool_log(
                    output.tool, output.tool_input, observation
                )
                self.intermediate_steps.append((output, observation))
                return message, False
        except BadRequestError as error:
            return self.handle_error(error), True

    def handle_error(self, error: BadRequestError) -> str:
        # コンテキスト長あふれの可能性もあるため、最後のステップのツール実行結果を空にする
        self.intermediate_steps[-1] = (self.intermediate_steps[-1][0], "")
        return self.fallback_chain.invoke(
            {
                "input": error.response.json()["error"]["message"],
                "intermediate_steps": self.intermediate_steps,
            }
        )

    def tool_execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        tool_dict = {tool.name: tool for tool in self.tools}
        selected_tool = tool_dict.get(tool_name)
        if selected_tool:
            return selected_tool.run(tool_input)
        else:
            raise ValueError(f"対応していないツールが選択されました: {tool_name}")

    def format_tool_log(
        self, tool_name: str, tool_input: Dict[str, Any], observation: str
    ) -> str:
        if tool_name == "read_file":
            result = f"{tool_input['file_path']}を読み込みました。\n" f"文字数：{len(observation)}\n"
        else:
            result = f"{observation}\n"

        return f"ツール選択\n" f"{tool_name}, {str(tool_input)}\n\n" f"実行結果\n" f"{result}"

    def finish_process(self, input_message: str, output_message: str) -> None:
        self.memory.save_context({"input": input_message}, {"output": output_message})
        self.intermediate_steps = []
