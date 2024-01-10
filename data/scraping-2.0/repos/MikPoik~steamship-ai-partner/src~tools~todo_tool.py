from steamship import Steamship #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task #upm package(steamship)
from steamship.agents.schema import AgentContext, Tool #upm package(steamship)
from steamship.agents.utils import get_llm, with_llm #upm package(steamship)

DEFAULT_PROMPT = """
Come up with a TODO list for task {input}

"""


class TodoTool(Tool):
    """
    Tool for generating Todo lists
    """

    name: str = "TodoTool"
    human_description: str = "Useful for generating a todo list"
    agent_description: str = (
        "Use this tool to generate a TODO list "
        "The input is the task to create a TODO list for"
        "The output is the generated TODO list"
        

    )
    rewrite_prompt: str = DEFAULT_PROMPT

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        llm = get_llm(with_llm(llm=OpenAI(client=context.client)))

        blocks = []
        for block in tool_input:
        # If the block is not text, simply pass it through.
            if not block.is_text():
                continue

            prompt = self.rewrite_prompt.format(input=block.text)
            output_blocks = llm.complete(prompt=prompt)
            blocks.extend(output_blocks)
        
        return output_blocks
        
if __name__ == "__main__":
    with Steamship.temporary_workspace() as client:
        ToolREPL(TodoTool()).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client, temperature=0.9))
        )
