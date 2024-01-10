from .ui_base import UIBase
from .role import Role
from .bot_agent import BotAgent
from .file_reader import FileReader
from .message_carrier import MessageCarrier
from .task import Task, TaskTag
from dotenv import load_dotenv
from .i18n import _
import asyncio
import openai
import os

class Session():
    """
    Autonomous task resolution sessions.
    """

    def __init__(self, view: UIBase) -> None:
        self.view = view
        self.message_carrier = MessageCarrier(view)
        self.file_reader = FileReader()
        # Read the API key from .env.
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        # Checks if the OpenAI API key has been set and returns an exception if not.
        if not openai.api_key:
            raise ValueError("APIKey is not set.")
        
    async def run(self) -> None:
        """
        Start Session.        
        """
        # 1. take an objective and its context
        # 2. split objective into tasks
        # 3. split tasks into resolvable resolvables and unresolvables
        # 4. based on unresolvables, summarize functions to be added and make need_function_tasks
        # 5. divide need_function_tasks into resolvables and unresolvables and distribute to each # 6.
        # 6. repeat 4-5 until unresolvables is empty
        # 7. resolve resolvables
        self.message_carrier.print_message_as_system("=== Start Session ===", True)
        objective, context = await self.determine_objective()
        tasks = self.split_to_tasks(objective, context)
        await self.end()

    async def end(self) -> None:
        self.message_carrier.print_message_as_system("=== End Session ===", True)
        self.message_carrier.save_log_as_json()
        # Display a message to exit when you type something.
        self.message_carrier.print_message_as_system(_("Enter something and it will exit."), False)
        await self.view.request_user_input()

    
    async def determine_objective(self) -> tuple[str, str]:
        """
        Ask the user for objectives until a feasible objective and its context are established.
        """
        while True:
            objective = await self.ask_objective()
            context = await self.ask_context()
            feasibility = await self.feasibility_assessment(objective, context)
            if not feasibility:
                text = _("Error: Unobtainable. \nPlease start over from the objective setting.")
                self.message_carrier.print_message_as_system(text, True)
            else:
                text = _("The objective has been determined to be achievable. \nGenerate task ......")
                self.message_carrier.print_message_as_system(text, True)
                return objective, context


    async def decide_on_specifications(self) -> str:
        """
        仕様についてユーザーと議論し、策定する。
        """
        pass

    def check_specifications_are_finalized(self) -> bool:
        """
        仕様が策定されているか確認する。
        """
        pass

    async def feasibility_assessment(self, objective: str, context: str) -> bool:
        """
        Determine feasibility of objectives.
        """
        agent = BotAgent()

        text = _("Determine the feasibility of your objectives ......")
        self.message_carrier.print_message_as_system(text, True)

        abilities = self.file_reader.read_file("abilities.txt", "documents")
        prompt = f"""
        You are an AI that determines if the objective given by the user are feasible.
        You are part of a repository called AutoEvolver.
        AutoEvolver is capable of: 
        {abilities}
        Based on the above, determine whether or not the following objective is feasible.
        Response with a just simple "Yes" or "No”.
        Objective: {objective}
        Context: {context}
        Response: 
        """

        agent.add_context(prompt)
        response = agent.response_to_context()
        self.view.process_event()
        if "Yes" in response:
            return True

        prompt = f"""
        Please tell me why you have determined that this task is not feasible.
        Response:"""
        agent.add_context(prompt)
        response = agent.response_to_context()
        self.view.process_event()
        # responseを解決不能な理由として表示する
        self.message_carrier.print_message_as_system(response, True)
        return False


    async def ask_objective(self) -> str:
        """
        Receive input from the user on the desired settings.
        """
        text = (_("Please enter a objective. \nExample: Please make a Tetris."))
        self.message_carrier.print_message_as_system(text, True)
        
        try:
            objective = await self.view.request_user_input()
            self.view.process_event()
            if not objective:
                raise ValueError(_("The objective has not been entered."))
        except asyncio.CancelledError:
            raise
        
        text = "Objective: " + objective
        self.message_carrier.print_message_as_user(text, True)

        return objective
    

    async def ask_context(self) -> str:
        """
        Receive contextual input from the user regarding the objective setting.
        """
        text = (_("Enter the context regarding the objective. \nExample: a simple Tetris that can be executed in Python. No sound is required."))
        self.message_carrier.print_message_as_system(text, True)

        try:
            context = await self.view.request_user_input()
            self.view.process_event()
        except asyncio.CancelledError:
            raise
        text = "Context: " + context
        self.message_carrier.print_message_as_user(text, True)

        return context
                

    def split_to_tasks(self, objective: str, context: str) -> list[Task]:
        agent = BotAgent()
        prompt = f"""
        You are an AI listing tasks to be performed based on the following objective: {objective}.
        The context regarding the objective is as follows: {context}
        You are part of a repository called AutoEvolver and this objective must be resolved by AutoEvolver alone.
        Subdivide and list the objectives into tasks in order to resolve them. Do not try to solve them at this point.
        When subdividing, do not add more than the original objective. Subdivide into tasks that require the least amount of effort to accomplish.
        If the task can be solved in the Python code implementation, subdivide it into modules.
        Do not create abstract tasks. Whenever possible, format the task to be solved by generating Python modules.
        The list should be formatted with the "-" sign and should not include responses other than the list.
        Response:"""
        response = agent.response(prompt)
        self.view.process_event()
        tasks_text = response.split("\n") if "\n" in response else [response]

        # Extract only lines starting with "-".
        tasks_text = [task for task in tasks_text if task.startswith("-")]

        # Convert all tasks_text to tasks
        tasks = [self.tasktext_to_task(objective, context, task_text) for task_text in tasks_text]

        # Display a list of Tasks before subdividing.
        # The list is displayed in order of Task's Content - TaskTag.value.
        # Messages are displayed in batches.
        self.message_carrier.print_message_as_system("=== Init Task List ===", True)
        print_text = ""
        for task in tasks:
            print_text += f"{task.content} - {task.tag.name}\n"
        self.message_carrier.print_message_as_system(print_text, True)

        # Check if each task should be subdivided, and set subtask if it should be subdivided.
        for task in tasks:
            if task.tag == TaskTag.subdivide:
                subtasks = self.split_to_subtasks(objective, context, task)
                task.subtasks = subtasks

        # Display the final list of Tasks.
        self.message_carrier.print_message_as_system("=== Confirmed Tasks ===", True)
        print_text = ""
        for task in tasks:
            print_text += f"{task.content} - {task.tag.name}\n"
            if task.subtasks:
                for subtask in task.subtasks:
                    print_text += f"    {subtask.content} - {subtask.tag.name}\n"
        self.message_carrier.print_message_as_system(print_text, True)

        return tasks
    
    def split_to_subtasks(self, objective: str, context: str, task: Task) -> list[Task]:
        """
        Split the Task into smaller Tasks and return a list of those Tasks.
        """
        agent = BotAgent()
        prompt = f"""
        You are an AI that further subdivides the subdivided tasks to achieve the final objective {objective}.
        The context of the objective is {context}.
        You are part of a repository called AutoEvolver.
        The task to subdivide is {task.content}.
        When subdividing, do not add more than the original objective. Subdivide into tasks that require the least amount of effort to accomplish.
        If the task can be solved in the Python code implementation, subdivide it into modules.
        The list should be formatted with the "-" sign and should not include responses other than the list.
        Response:"""
        response = agent.response(prompt)
        self.view.process_event()
        tasks_text = response.split("\n") if "\n" in response else [response]
        # Extract only lines beginning with "-".
        tasks_text = [task for task in tasks_text if task.startswith("-")]

        # Convert all tasks_text to tasks
        tasks = [self.tasktext_to_task(objective, context, task_text) for task_text in tasks_text]

        # Display a list of subdivided Tasks.
        self.message_carrier.print_message_as_system("=== Subdivided Tasks ===", True)
        print_text = ""
        for task in tasks:
            print_text += f"{task.content} - {task.tag.name}\n"
        self.message_carrier.print_message_as_system(print_text, True)

        return tasks


    def tasktext_to_task(self, objective: str, context: str, task_text: str) -> Task:
        agent = BotAgent()
        prompt = f"""
        You are an AI that categorizes the solution to a given task as "ask user (1)", "Further divide into smaller tasks (2)" "output text by ChatGPT itself (3)", "run or write new Python module to solve (4)" or "unsolvable (0)".
        The task is part of the final objective {objective}. The context of that objective is {context}.
        The task is: {task_text}
        Select only one most appropriate number and return that number only. Any other response will not be included.
        If it is difficult to determine, just answer "2" for now.
		Choose the solution with the highest number possible.
        Number:"""
        response = agent.response(prompt)
        self.view.process_event()
        try:
            digit = response[0]
            number = int(digit)
        except ValueError:
            raise ValueError(f"response is not a number: {response} for {task_text}")
        # responseからタグを抽出する
        tag_dict = {
            0: TaskTag.unsolvable, 
            1: TaskTag.ask_user, 
            2: TaskTag.use_bot, 
            3: TaskTag.use_python,
            4: TaskTag.subdivide,
            }
        tag = tag_dict[number]
        task = Task(task_text, tag)
        return task
    

    def resolve_tasks(self, agent: BotAgent, objective: str, context: str, tasks: list[Task]) -> None:
        pass


    def resolve_with_python(self, agent: BotAgent, objective: str, context: str, task: Task) -> None:
        pass