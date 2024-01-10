from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from langchain.schema import AgentAction, AIMessage, HumanMessage
from llama_index.response.schema import Response
from mdutils.mdutils import MdUtils

from xecretary_core.utils.utility import sep_md


@dataclass(frozen=True)
class ConversationLog:
    input: str
    output: str
    intermediate_steps: Union[Sequence[AgentAction], Sequence[str], None]
    chat_history: Optional[List[Union[HumanMessage, AIMessage]]]
    elapsed_time: float

    def dump(self, md_file: MdUtils) -> None:
        md_file.new_header(
            level=2, title=f"質問: {self.input}", add_table_of_contents="n"
        )

        md_file.new_line(f"実行時間: `{self.elapsed_time}`")

        if self.intermediate_steps:
            for step in self.intermediate_steps:
                (agent_action, answer) = step
                if isinstance(answer, Response):
                    answer = answer.response

                md_file.new_line(f"tool: `{agent_action.tool}`")
                md_file.new_line(f"tool input: `{agent_action.tool_input}`")
                md_file.new_line("log:")
                md_file.insert_code(agent_action.log, language="bash")
                md_file.new_line("answer: ")
                md_file.insert_code(answer, language="bash")
                sep_md(md_file)

        # We are not currently using chat_history as interactive utilization
        # is not assumed at present.
        """
        elif self.chat_history:
            for chat in self.chat_history:
                if isinstance(chat, HumanMessage):
                    md_file.new_line(f"human message: `{chat.content}`")
                elif isinstance(chat, AIMessage):
                    md_file.new_line(f"AI message: `{chat.content}`")

        md_file.new_line("final answer:")
        md_file.insert_code(self.output, language="bash")
        md_file.new_line()
        """
