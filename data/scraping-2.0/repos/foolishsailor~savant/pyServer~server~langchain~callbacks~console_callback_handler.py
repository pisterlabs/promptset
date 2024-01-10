import json
from yachalk import chalk
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction, AgentFinish

from typing import List, Dict, Union, Any


class ConsoleCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(chalk.blue("========= LLM Start ========="))
        print(chalk.green.bold("Serialized: "), json.dumps(serialized, indent=2))
        print(chalk.green.bold("Prompts: "), json.dumps(prompts, indent=2))
        print(chalk.green.bold("Other Args: "), kwargs)

    # def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
    #     print(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(chalk.blue("========= LLM End ========="))
        print(response)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print(chalk.red("========= LLM Error ========="))
        print(json.dumps(error, indent=2))

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        print(chalk.blue("========= Chain Start ========="))
        print(chalk.green.bold("Serialized: "), json.dumps(serialized, indent=2))
        print(chalk.green.bold("Inputs: "), json.dumps(inputs, indent=2))
        # print(chalk.green.bold("Other Args: "), json.dumps(kwargs, indent=2))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        print(chalk.blue("========= Chain End ========="))
        print(chalk.green.bold("Output: "), json.dumps(outputs, indent=2))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print(chalk.red("========= Chain Error ========="))
        print(json.dumps(error, indent=2))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        print(chalk.blue("========= Tool Start ========="))
        print(chalk.green.bold("Serialized: "), json.dumps(serialized, indent=2))
        print(chalk.green.bold("Input: "), input_str)
        print(chalk.green.bold("Other Args: "), json.dumps(kwargs, indent=2))

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        print(chalk.blue("========= Tool End ========="))
        print(chalk.green.bold("output: "), output)
        print(chalk.green.bold("Other Args: "), json.dumps(kwargs, indent=2))

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        print(chalk.red("========= Tool Error ========="))
        print(json.dumps(error, indent=2))

    def on_text(self, text: str, **kwargs: Any) -> Any:
        print(chalk.blue("========= On Text ========="))
        print(chalk.green.bold("Other Args: "), json.dumps(kwargs, indent=2))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        print(chalk.blue("==== Agent Action ===="))
        print(f"{chalk.magenta('  Agent Action RAW:')} {action}")
        print(f"{chalk.green.bold('    Agent Tool:')} {action.tool}")
        print(
            f"{chalk.green.bold('   Agent Input:')} {json.dumps(action.tool_input, indent=2)}"
        )
        print(f"{chalk.green.bold('     Agent Log:')} {action.log}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        print(chalk.blue("==== Agent Finish ===="))
        print(f"{chalk.magenta('  Agent Action RAW:')} {finish}")
        print(
            f"{chalk.green.bold('    Agent Tool:')} {json.dumps(finish.return_values, indent=2)}"
        )
        print(f"{chalk.green.bold('   Agent Input:')} {finish.log}")
