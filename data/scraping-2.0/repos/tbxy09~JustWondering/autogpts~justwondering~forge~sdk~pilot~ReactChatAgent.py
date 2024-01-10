from instructor.validate_args import validate_args
from pydantic import Field, BaseModel
from forge.sdk.pilot.Output import BaseOutput
from forge.sdk.pilot.CodeImp import CodeMonkeyRefactoredArgs
from forge.sdk.pilot.utils.shell import RunShellArgs
from typing import List, Union, Optional, Type, Dict, Callable, Tuple
import openai
import os
import json
import enum
import logging
import re

from forge.sdk.pilot.util import format_plugin_schema
from forge.sdk.pilot.utils.basetool import BaseTool
from forge.sdk.pilot.utils.base_agent import BaseAgent
from forge.sdk.pilot.utils.agent_model import AgentType, AgentOutput
from forge.sdk.pilot.utils.prompt_template import PromptTemplate
from forge.sdk.pilot.utils.base import AgentFinish, AgentAction
from pydantic import create_model, BaseModel
from pygments import highlight, lexers, formatters
import instructor


class ChatCompletionArgs(BaseModel):
    engine: str
    messages: list[dict]
    response_model: Optional[Type[BaseModel]] = None
    function_call: Optional[Dict] = None
    functions: Optional[List[Callable]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None


class Tool(enum.Enum):
    CodeMonkey = "CodeMonkey"
    RunShell = "bash_shell"


class TalksEntity(BaseModel):
    # id: int = Field(..., description="id of the action")
    developer: str = Field(...,
                           description="extract the the developer part of the conversation, ")
    coding_worker: str = Field(...,
                               description="extract the the coding worker part of the conversation, ")
    # tool: Tool = Field(..., description="tool to use to do the action")
    # tool_input: Union[RunShellArgs,
    #   CodeMonkeyRefactoredArgs] = Field(..., description="input to the tool")
    # questions: str = Field(
    #     None, description="question from coding worker for unclear parts, if no, just say no more questions")
    # dependencies: List[int] = Field(
    #     ..., description="list of action ids that this action depends on")


class ActionItem(BaseModel):
    id: int = Field(..., description="id of the action")
    tool: Tool = Field(..., description="tool to use to do the action")
    tool_input: Union[RunShellArgs,
                      CodeMonkeyRefactoredArgs] = Field(..., description="input to the tool")
    dependencies: List[int] = Field(
        ..., description="list of action ids that this action depends on")


class FinalAnswerEntity(BaseModel):
    output: Optional[str] = Field(
        ..., description="if it is a final state, text summary of the final answer")
    ActionList: List[ActionItem] = Field(
        ..., description="list of actions with order")


class ResponseEntity(BaseModel):
    action: Union[TalksEntity, FinalAnswerEntity] = Field(
        ..., description="state of talks between developer and coding worker, still in discussion or reach a final answer state, no unclear parts")


# print(ResponseEntity.schema_json(indent=2))
class User(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    role: str = Field(description="The role of the person")


# user = User(name="Jason Liu", age=20, role="student")
# logger = PathLogger(__name__)
# logger.debug(json.dumps(ResponseEntity.model_json_schema(), indent=2))
# project = Project({})
# agent = Agent('user', project)
# system_message = ""
# system_message += "Any Instruction you get which labeled as **IMPORTANT**, you follow strictly."

# convo = AgentConvo(agent,system_message)
FINAL_ANSWER_ACTION = "Final Answer:"
output = BaseOutput()
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


class ReactChatAgent(BaseAgent):
    name: str = "ReactAgent"
    type: AgentType = AgentType.react
    version: str
    description: str
    # target_tasks: list[str]
    prompt_template: PromptTemplate
    respone: object
    plugins: List[Union[BaseTool, BaseAgent]]
    examples: Union[str, List[str]] = None  # type: ignore
    args_schema: Optional[Type[BaseModel]] = create_model(
        "ReactArgsSchema", instruction=(str, ...))
    logger: Optional[object]

    intermediate_steps: List[Tuple[AgentAction, str]] = []
    intermediate_steps_index: int = 0

    raw_message: List[str] = []

    def set_logger(self, logger):
        self.logger = logger

    def artifact_handler(self):
        self.logger.info("artifact_handler")
        return

    # server agent api, hand to agent to wrap the response for a agent protocol
    def api_task(self, input):
        self.logger.info(f"api_task: {input}")
        # self.run(input)
        self.get_final_answer()
        return

    def api_step(self, step):
        self.logger.info(f"api_step: {step}")
        self.run(step)
        return

    def _compose_plugin_description(self) -> str:
        """
        Compose the worker prompt from the workers.

        Example:
        toolname1[input]: tool1 description
        toolname2[input]: tool2 description
        """
        prompt = ""
        try:
            for plugin in self.plugins:
                prompt += f"{plugin.name}: {plugin.description}\n"
        except Exception:
            raise ValueError("Worker must have a name and description.")
        return prompt

    def _construct_raw_message(self, raw: List[str]) -> str:
        return "\n".join(raw)

    def _construct_action_scratchpad(
            self, intermediate_steps: List[Tuple[TalksEntity, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        # for action, justdoit, observation, sufix in intermediate_steps:
        for action in intermediate_steps:
            #     if action.questions is None or action.questions == "":
            #         action.questions = "no more questions"
            thoughts += f"\nDeveloper:"
            thoughts += action.developer
            thoughts += f"\nCoding worker: {action.coding_worker}"
            # thoughts += f"\nCoding worker: {action.coding_worker} \nQuestions:"
            # thoughts += f"{action.questions}\nDeveloper:"
            # thoughts += f"\nAction Input: {action.tool_input}"
        # replace <content> with content
        # thoughts = re.sub(r"<(.*?)>", r"\1", thoughts)
        return thoughts

    def _construct_scratchpad(
            self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        # for action, justdoit, observation, sufix in intermediate_steps:
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought:"
        # replace <content> with content
        # thoughts = re.sub(r"<(.*?)>", r"\1", thoughts)
        return thoughts

    def _parse_output(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise Exception(
                    "Parsing LLM output produced both a final answer "
                    f"and a parse-able action: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise Exception(
                f"Could not parse LLM output: `{text}`",
            )
        elif not re.search(
                r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise Exception(
                f"Could not parse LLM output: `{text}`"
            )
        else:
            raise Exception(f"Could not parse LLM output: `{text}`")

    def _compose_condition_prompt(self, instruction) -> str:
        """
        Compose the prompt from template, worker description, examples and instruction.
        """
        agent_scratchpad = self._construct_scratchpad(self.intermediate_steps)
        tool_description = self._compose_plugin_description()
        tool_names = ", ".join([plugin.name for plugin in self.plugins])
        # if self.prompt_template is None:
        self.prompt_template = PromptTemplate(
            input_variables=["instruction", "agent_scratchpad",
                             "tool_names", "tool_description"],
            template="""
Action 1: Tool A
If success:
  Action 2: Tool B
Else: 
  Action 2: Tool C
"""
        )

    def _compose_prompt(self, instruction) -> str:
        """
        Compose the prompt from template, worker description, examples and instruction.
        """
        # agent_scratchpad = self._construct_action_scratchpad(
        #     self.intermediate_steps)
        agent_scratchpad = self._construct_raw_message(self.raw_message)
        tool_description = self._compose_plugin_description()
        tool_names = ", ".join([plugin.name for plugin in self.plugins])
        # if self.prompt_template is None:
        self.prompt_template = PromptTemplate(
            input_variables=["instruction", "agent_scratchpad",
                             "tool_names", "tool_description"],
            # current file path
            template=open(os.path.join(os.path.dirname(__file__),
                          "prompts/talks_dev_coder.md"), "r").read()
        )
#         self.prompt_template = PromptTemplate(
#             input_variables=["instruction", "agent_scratchpad",
#                              "tool_names", "tool_description"],
#             template="""Answer the following questions as best you can. You have access to the following tools:
# {tool_description}.
# Use the following format:
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
# Begin!
# Question: {instruction}
# Thought:{agent_scratchpad}
#     """)
        return self.prompt_template.format(
            instruction=instruction,
            agent_scratchpad=agent_scratchpad,
            tool_description=tool_description,
            tool_names=tool_names
        )

    def _format_function_map(self) -> Dict[str, Callable]:
        # Map the function name to the real function object.
        function_map = {}
        for plugin in self.plugins:
            if isinstance(plugin, BaseTool):
                function_map[plugin.name] = plugin._run
            else:
                function_map[plugin.name] = plugin.run
        return function_map

    def _format_multi_plugins_schema(self, plugins: List[Union[BaseTool, BaseAgent]]):
        """Format list of tools into the open AI function API.

        :param plugins: List of tools to be formatted.
        :type plugins: List[Union[BaseTool, BaseAgent]]
        :return: Formatted tools.
        :rtype: Dict
        """
        schema = {
            'type': 'object',
            'properties': {
                "needs_plan_or_replan": {
                    "type": "boolean",
                    "description": "whether the task needs a plan or replan",
                },
                "tools": {
                    'type': 'array',
                    'description': 'List of actions item that need to be done to complete the task. if a replan is needed, the list will only contain remaining actions which is fail or unknown.',
                    'items': {
                        'type': 'object',
                        'description': 'one of the action items that need to be done to complete the task.',
                        'properties': {
                            'name': {
                                'type': 'string',
                                'enum': [plugin.name for plugin in plugins],
                                'description': 'Name of the tool.',
                            },
                        }
                    },
                }
            }
        }
        for (i, plugin) in enumerate(plugins):
            schema['properties']['tools']['items']['properties'][plugin.name] = plugin.args_schema.schema()
        return {
            'name': 'process_tools',
            'description': 'Process list of tools',
            'parameters': schema,
        }

    def _response_from(self, response: Dict):
        # message = response["choices"][0]["message"]
        message = response
        # assert "function_calls" in message, "No function call detected"
        if "function_calls" not in message:
            func_name = self.intermediate_steps[self.intermediate_steps_index][0].tool
            func_args = self.intermediate_steps[self.intermediate_steps_index][0].tool_input
            self.intermediate_steps[-1][-1] = (
                self._format_function_map()[func_name](**func_args))
            self.intermediate_steps_index += 1
            return
        # remove items after self.intermediate_steps_index
        while len(self.intermediate_steps) > self.intermediate_steps_index and len(self.intermediate_steps) > 0:
            self.intermediate_steps.pop()
            # self.intermediate_steps[self.intermediate_steps_index][-1] = "throw the plan away"
            # self.intermediate_steps[self.intermediate_steps_index][-2] = "no observation"
            # self.intermediate_steps[self.intermediate_steps_index].remove_last_2_messages()
            # self.intermediate_steps[self.intermediate_steps_index].pop()
            # self.intermediate_steps[self.intermediate_steps_index].pop()
            # self.intermediate_steps_index += 1
        # self.intermediate_steps.remove(self.intermediate_steps_index)
        function_calls = message["function_calls"]['arguments']['tools']
        func = function_calls[0]
        func_name = func["name"]
        func_args = func[func_name]
        self.intermediate_steps.append([AgentAction(
            func["name"], func_args, f"function call: {func['name']} with arguments: {func_args}")])
        self.intermediate_steps[-1].append(
            f"function call: {func['name']} with arguments: {func_args}")
        self.intermediate_steps[-1].append(self._format_function_map()[
                                           func["name"]](**func_args))
        self.intermediate_steps_index += 1
        # output.panel_print(f"Action: {func['name']}")
        # output.panel_print(f"Tool Input: {func_args}")
        self.intermediate_steps[-1].append('DONE, not give the same plan')
        # logging.info(f"Tool Input: {func['arguments']}")
        for func in function_calls[1:]:
            func_name = func["name"]
            func_args = func[func_name]
            self.intermediate_steps.append([AgentAction(
                func["name"], func_args, f"function call: {func['name']} with arguments: {func_args}")])
            self.intermediate_steps[-1].append("")
            self.intermediate_steps[-1].append("")
            self.intermediate_steps[-1].append("")
            # output.panel_print(f"Action: {func['name']}")
            # output.panel_print(f"Tool Input: {func_args}")
        return

    def get_prompt(self, template, data=None):
        from jinja2 import Environment, FileSystemLoader
        if data is None:
            data = {}
        # step_statuses = {'step_1': 'failed'}
        jinja_env = Environment()
        rendered = jinja_env.from_string(
            template).render(data)
        return rendered

    def _format_function_schema(self) -> List[Dict]:
        # List the function schema.
        function_schema = []
        for plugin in self.plugins:
            if isinstance(plugin, BaseTool):
                function_schema.append(plugin.schema)
            else:
                function_schema.append(plugin.schema)
        return function_schema

    def save_checkpoint(self, args):
        # dump self.intermediate_steps to path
        import json
        path = "checkpoint.json"
        with open(path, 'w') as f:
            if len(self.intermediate_steps) > 0:
                json.dump(self.intermediate_steps, f)
            else:
                json.dump(args, f)

    def pretty_print(self, json_object):
        json_str = json.dumps(json_object, indent=2)
        self.logger.debug(
            highlight(json_str, lexers.JsonLexer(), formatters.TerminalFormatter()))

    def send_msg(self, args):
        # Validate args
        args.dict()
        return openai.ChatCompletion.create(**args.dict())

    def run(self, instruction, max_iterations=10):
        """
        Run the agent with the given instruction.

        :param instruction: Instruction to run the agent with.
        :type instruction: str
        :param max_iterations: Maximum number of iterations of reasoning steps, defaults to 10.
        :type max_iterations: int, optional
        :return: AgentOutput object.
        :rtype: AgentOutput
        """
        self.clear()
        logging.info(
            f"Running {self.name + ':' + self.version} with instruction: {instruction}")
        total_cost = 0.0
        total_token = 0
        from forge.sdk.pilot.util import save_prompt_to_file
        FUNC_CALL_LIST = {
            "definitions": [
            ],
            "function_calls": self._format_function_map()
        }
        for plugin in self.plugins:
            if isinstance(plugin, Union[BaseTool, BaseAgent]):
                FUNC_CALL_LIST["definitions"].append(
                    format_plugin_schema(plugin))
            else:
                # throw Exception("Not support agent yet")
                raise Exception("Not support agent yet")
                # FUNC_CALL_LIST["definitions"].append(plugin.schema())
                # FUNC_CALL_LIST["function_calls"] += plugin.name + "(" + plugin.name + "_args),"
        # print(FUNC_CALL_LIST)
        for _ in range(max_iterations):
            # for _ in range(1):
            prompt = self._compose_prompt(instruction)
            workfolder = os.path.dirname(
                os.path.abspath(__file__))
            prompt_path = os.path.join(
                workfolder, "prompts/execute_commands.prompt")
            save_prompt_to_file(prompt_path=prompt_path, prompt_content=prompt)
            ret = None
            try:
                with open("checkpoint.json", 'r') as f:
                    ret = json.load(f)
            except Exception:
                ret = None
            ignore_checkpoints = True
            if ret == None or len(ret) == 0 or ignore_checkpoints:
                openai.api_type = "azure"
                openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
                openai.api_version = "2023-07-01-preview"
                openai.api_key = os.getenv("AZURE_API_KEY")
                deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                workspace = os.environ["AGENT_WORKSPACE"]
                response = None
                tool_description = self._compose_plugin_description()
                try:
                    args = {
                        'engine': deployment_name,
                        'temperature': 0.9,
                        # 'response_model': [ActionItem,FinalAnswerEntity],
                        'messages': [
                            {"role": "system", "content": ""},
                            {"role": "assistant",
                                "content": ""
                             },
                            # {"role": "system", "content": f"{instruction}"},
                            # {"role": "assistant", "content": f""},
                            {"role": "user", "content": prompt},
                            # {"role": "user", "content": "explain the your plan and answer the question if has any and continue the talks imitatation, if no more questions just reponse 'No more question' and coding worker summarize an action list using yaml description."}
                            # {"role": "user", "content":"any questions?"},
                        ],
                        # 'stop': ['\nDeveloper:', '\n\tDeveloper:']
                        'stop': ['No more questions', 'no more questions', 'Final Answer']
                    }
                    ChatCompletionArgs(**args)
                    response = openai.ChatCompletion.create(
                        **args)
                except Exception as e:
                    if e.__class__.__name__ == "ValidationError":
                        str_items = e.errors()[0]['loc']
                        error = f"field '{e.errors()[0]['loc'][-1]}' from {'.'.join(str_items[:-1])}"
                        print(f" {e.errors()[0]['type']}: {error}")
                    raise e

            self.logger.debug(response["choices"][0]["message"]["content"])
            self.raw_message.append(
                response["choices"][0]["message"]["content"])
            # content = self._construct_raw_message(self.raw_message)
            continue
            # self.logger.debug(self._format_function_map())

            action = response.action
            if isinstance(action, FinalAnswerEntity):
                # print(f"get final answer: {action.output}")
                self.logger.info(action.model_dump_json())
                break
            # tool_name = action.tool.value
            # tool_explanation = action.explanation
            self.intermediate_steps.append(action)
            # if isinstance(tool_input, BaseModel):
            #     tool_input = tool_input.model_dump()
            #     result = self._format_function_map()[tool_name](**tool_input)
            # if isinstance(tool_input, dict):
            #     result = self._format_function_map()[tool_name](**tool_input)
            # else:
            #     result = self._format_function_map()[tool_name](tool_input)
            # if isinstance(result, AgentOutput):
            #     total_cost += result.cost
            #     total_token += result.token_usage
            # if isinstance(result, str):
            #     self.intermediate_steps.append(
            #         (AgentAction(tool_name, tool_input, tool_explanation), ""))
            # else:
            #     self.intermediate_steps.append(
            #         (AgentAction(tool_name, tool_input, tool_explanation), ""))
        return None

    def get_final_answer(self):
        with open("prompts/execute_commands_cp2.prompt", 'r') as f:
            self.raw_message = f.readlines()
        instructor.patch()
        content = self._construct_raw_message(self.raw_message)
        self.logger.debug(content)
        args = {
            'engine': deployment_name,
            'temperature': 0.9,
        }
        args['response_model'] = FinalAnswerEntity
        args['messages'] = [{
            "role": "system",
            "content": ""
        },
            {
            "role": "user",
            "content": f"extract action item from {content}"
        }
        ]
        response: FinalAnswerEntity = openai.ChatCompletion.create(**args)
        instructor.unpatch()
        self.logger.debug(response.model_dump_json())
        quit()
        # with open("response.json", 'w') as f:
        #     json.dump(response.model_dump_json(), f)
        for ac in response.ActionList:
            result = self._format_function_map()[ac.tool.value](
                **ac.tool_input.model_dump())
            self.logger.debug(result)

    def stream(self, instruction: Optional[str] = None, output: Optional[BaseOutput] = None, max_iterations: int = 10):
        pass

    def clear(self):
        """
        Clear and reset the agent.
        """
        self.intermediate_steps.clear()
