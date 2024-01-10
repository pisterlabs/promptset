from typing import Callable, Dict, List, Optional
from openai.types.chat import ChatCompletionMessageToolCall

from democratic_agent.architecture.helpers import Request
from democratic_agent.chat.parser.pydantic_parser import PydanticParser
from democratic_agent.chat.chat import Chat
from democratic_agent.tools.tools_manager import ToolsManager
from democratic_agent.utils.helpers import colored


MAX_ITERATIONS = 30  # TODO: Move to config.


class Execution:
    def __init__(self, summary: str, success: bool):
        self.summary: str = summary
        self.success: bool = success


# TODO: Set MAX short term memory size and MAX conversation size!(Important to calculate number of tokens).
class Executor:
    """Execute different tools to satisfy the user request."""

    def __init__(self, get_user_feedback: Callable, user_name: str):
        self.chat = Chat(
            module_name="executor",
            system_prompt_kwargs={"request": ""},
            user_name=user_name,
        )
        self.tools_manager = ToolsManager()
        self.planner_functions = [
            self.find_tools,
            self.select_tool,
            self.search_user_info,
            self.ask_user,
            self.set_task_completed,
        ]
        self.selected_tools: Dict[str, Callable] = {}
        self.get_user_feedback = get_user_feedback
        self.new_plan = None

    def get_response(
        self, feedback: Optional[str]
    ) -> List[ChatCompletionMessageToolCall]:
        print(
            colored("\n--- Planner ---\n", "cyan")
        )  # TODO: Make this part of logger with different colors depending on the running module of chat..
        if feedback:
            planner_prompt_kwargs = {
                "feedback": feedback,
            }
            return self.chat.get_response(
                prompt_kwargs=planner_prompt_kwargs, functions=self.planner_functions
            )
        return

    def execute(self, request: Request) -> Execution:
        self.request = request
        self.chat.edit_system_message(system_prompt_kwargs={"request": request.request})
        self.chat.conversation.add_user_message(
            "New task received on System message. Start working towards solving it."
        )

        iterations = 0
        self.execution = None

        # TODO: Send iterations to the prompt? This can be seen as a "frustration" metric that maybe helps the model to take more risks.
        while iterations < MAX_ITERATIONS:
            print(colored(f"\n--- STEP {iterations}---\n", "blue"))
            functions = self.planner_functions.copy()
            functions.extend(self.selected_tools.values())

            tools_call = self.chat.call(functions=functions)
            try:
                if tools_call is not None:
                    if isinstance(tools_call, str):
                        print(f"Tools call is a string: {tools_call}")
                        return Execution(
                            summary=f"Failed to execute tools as model returned a string with content: {tools_call}",
                            success=False,
                        )
                    else:
                        tools_result = self.tools_manager.execute_tools(
                            tools_call=tools_call,
                            functions=functions,
                            chat=self.chat,
                        )

                        # Remove selected tool if already executed.
                        for tool_resut in tools_result:
                            if self.selected_tools.get(tool_resut.name):
                                self.selected_tools.pop(tool_resut.name)

                        # Check if we have plan or task is completed.
                        if self.execution:
                            return self.execution
            except Exception as e:
                print(f"Error executing tools: {e}")
            iterations += 1
        print(
            colored(
                "Max iterations reached... returning empty plan.",
                "red",
            )
        )
        return Execution(
            summary="Max iterations reached... returning empty plan.",
            success=False,
        )

    def select_tool(self, step_description: str, tool_name: str):
        """
        Select a specific tool that should be added in the prompt to be used in the next iteration.

        Args:
            step_description (str): The description of the step that should be accomplished using the tool.
            tool_name (str): The name of the tool that should be used to accomplish the step.

        Returns:
            str: The description of the step that should be accomplished next.
            str: The name of the tool that will be executed.
        """
        # Call the model to get the tool args.
        try:
            callable_function = self.tools_manager.get_tool(tool_name)
            self.selected_tools[tool_name] = callable_function
            return f"Tool {tool_name} added to prompt."
        except Exception as e:
            return f"Error calling tool: {e}"

    def find_tools(self, potential_approach: str, descriptions: List[str]):
        """
        Description of hypothetical tools that could be used to solve the current step.

        Args:
            potential_approach (str): The potential approach that could be used to solve the current step.
            descriptions (List[str]): The descriptions of the tools that could be used.

        Returns:
            callable: The tool that was created.
        """
        potential_tools: Dict[str, str] = {}
        for description in descriptions:
            tool = self.chat.database.search_tool(description)
            if tool is not None:
                if tool["name"] not in potential_tools.keys():
                    potential_tools[tool["name"]] = tool["description"]
        if not potential_tools:
            return f"No tools found for approach: {potential_approach}"
        tools_str = "\n".join(
            f"Tool: {name}, description: {description}"
            for name, description in potential_tools.items()
        )
        return f"Found available tools:\n{tools_str}"

    def ask_user(self, message: str):
        """
        Ask something to the user. Useful when you don't find critical information to solve the request.

        Args:
            message (str): The message to be sent to the user.

        Returns:
            callable: The tool that was created.
        """
        # Send message
        print(colored("System requires feedback: ", "red") + message)

        self.request.update_feedback(feedback=message)
        response = self.get_user_feedback(self.request)
        return f"User response: {response}"

    def set_task_completed(self, summary: str, success: Optional[bool] = True):
        """
        Set the task as completed, add the summary and if the task was completed successfully.

        Args:
            summary (str): The summary of the task.
            success (bool): If the task was completed successfully
        """
        self.execution = Execution(summary=summary, success=success)

        return "Task completed."

    def search_user_info(self, user_info: str):
        """
        Search user info.

        Args:
            user_info (str): The user info to be searched.

        Returns:
            str: The user info.
        """
        # THIS MEANS THAT SYSTEM DB IS THE SAME AS USER DB, MODIFY IF WE CENTRALIZE USER DB (1 db for user and multiple systems connected).
        data = self.chat.search_on_long_term_memory(user_info)
        if data is None:
            return "Information not found."
        return data

    def register_tools(self):
        """
        Register tools to the tool manager.

        Returns:
            str: The summary of the task.
        """
        tool_names = self.tools_manager.get_all_tools()
        tools: List[Callable] = []
        for tool_name in tool_names:
            tools.append(self.tools_manager.get_tool(tool_name))
        for tool in tools:
            function_schema = PydanticParser.get_function_schema(tool)["function"]
            self.chat.database.store_tool(
                name=function_schema["name"], description=function_schema["description"]
            )
