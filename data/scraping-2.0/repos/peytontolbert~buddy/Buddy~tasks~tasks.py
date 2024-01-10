import json
import openai
import time
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from typing import List, Any
from langchain import LLMChain
from llm.generate_task_plan.prompt import get_template
from llm.list_output_parser import LLMListOutputParser


class Task(BaseModel):
    """Task model."""
    id: int = Field(..., description="Task ID")
    description: str = Field(..., description="Task description")
    is_done: bool = Field(False, description="Task done or not")
    result: str = Field("", description="The result of the task")


class TaskManager(BaseModel):
    """Task manager model."""
    tasks: List[Task] = Field([], description="The list of tasks")
    current_task_id: int = Field(1, description="The last task id")

    def generate_task_plan(self, message: str, retries=5, delay=5):
        """Generate a task plan for the agent."""      
        BASE_TEMPLATE = """
        You should create a task that uses the result of an execution agent
        to create a new task with the following GOAL:

        [MESSAGE FROM THE AGENT]
        {message}

        [YOUR MISSION]
        Based on the [THOUGHTS], create new task to be completed by the AI system that do not overlap with incomplete tasks.
        - You can create any number of tasks.

        [RESPONSE FORMAT]
        Return the tasks as a list of string.
        - Enclose each task in double quotation marks.
        - Separate tasks with Tabs.
        - Reply in first-person.
        - Use [] only at the beginning and end

        ["Task 1 that I should perform"\t"Task 2 that I should perform",\t ...]

        [RESPONSE]
        """
        chat_input = BASE_TEMPLATE.format(message=message)
        
        for i in range(retries):
            try:
                
                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
                result =  str(results['choices'][0]['message']['content'])
                print(result)
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise


            # Parse and validate the result
            try:
                result_list = LLMListOutputParser.parse(result, separeted_string="\t")
            except Exception as e:
                raise Exception("Error: " + str(e))

            # Add tasks with a serial number
            for i, e in enumerate(result_list, start=1):
                id = int(i)
                description = e
                self.tasks.append(Task(id=id, description=description))

            self

    def get_task_by_id(self, id: int) -> Task:
        """Get a task by Task id."""
        for task in self.tasks:
            if task.id == id:
                return task
        return None

    def get_current_task(self) -> Task:
        """Get the current task agent is working on."""
        return self.get_task_by_id(self.current_task_id)

    def get_current_task_string(self) -> str:
        """Get the current task agent is working on as a string."""
        task = self.get_current_task()
        if task is None:
            return None
        else:
            return self._task_to_string(task)

    def eval_action(self, action_result_string: str, current_task: None) -> bool:
        """Evaluate an action."""
        prompt = """You are an intelligent agent. You should evaluate and decide if the recent action completed the task or not.:
        [RECENT ACTION]
        {action}

        [CURRENT TASK]
        {task}

        [YOUR MISSION]
        Based on the recent action, your job is to respond whether the current task has been completed or not.
        [RESPONSE FORMAT]
        Return "Yes" or "No" as a string. The question is has the current task been complete?

        [RESPONSE]
        """
        chat_input = prompt.format(action=action_result_string, task=current_task)
        print(chat_input)
        retries, delay = 20, 5     
        for i in range(retries):
            try:
                
                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": chat_input}])
                result =  str(results['choices'][0]['message']['content'])
                print(result)

                if result.lower() == "yes":
                    self.complete_current_task(result=action_result_string)
                elif result.lower() == "no":
                    return
                else:
                    if i<retries-1:
                        time.sleep(delay)
                    else:
                        raise ValueError("Invalid response from OpenAI API")
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def complete_task(self, id: int, result: str) -> None:
        """Complete a task by Task id."""
        # Complete the task specified by ID
        if id > 0 and id <= len(self.tasks):
            self.tasks[id - 1].is_done = True
            self.tasks[id - 1].result = result
            self.current_task_id += 1
        else:
            print(f"Task with id {id} does not exist")

    def complete_current_task(self, result: str) -> None:
        """Complete the current task agent is working on."""
        self.complete_task(self.current_task_id, result=result)

    def _task_to_string(self, task: Task) -> str:
        """Convert a task to a string."""
        return f"{task.id}: {task.description}"

    def get_incomplete_tasks(self) -> List[Task]:
        """Get the list of incomplete tasks."""
        return [task for task in self.tasks if not task.is_done]

    def get_incomplete_tasks_string(self) -> str:
        """Get the list of incomplete tasks as a string."""
        result = ""
        for task in self.get_incomplete_tasks():
            result += self._task_to_string(task) + "\n"
        return result
    def modify_current_task(self, thought):
        prompt = """You have attempted this task three times, and need to modify the task to make it easier to complete.
        [FULL REQUEST]
        {thought}
        
        
        
        [CURRENT TASK LIST]
        {tasks}

        [RESPONSE FORMAT]
        Return the tasks as a list of string.
        - Enclose each task in double quotation marks.
        - Separate tasks with Tabs.
        - Reply in first-person.
        - Use [] only at the beginning and end

        ["Task 1 that I should perform"\t"Task 2 that I should perform",\t ...]

        [RESPONSE]
        """
        retries=15
        delay=10
        newprompt = prompt.format(thought=thought, tasks=self.tasks)       
        for i in range(retries):
            try:
                
                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": newprompt}])
                result =  str(results['choices'][0]['message']['content'])
                print(result)
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise


            # Parse and validate the result
            try:
                result_list = LLMListOutputParser.parse(result, separeted_string="\t")
            except Exception as e:
                raise Exception("Error: " + str(e))

            # Add tasks with a serial number
            for i, e in enumerate(result_list, start=1):
                id = int(i)
                description = e
                self.tasks.clear()
                self.tasks.append(Task(id=id, description=description))

            self

        return  self.get_current_task_string()
    
    def clarify (self, message:str ) -> str:
        prompt = """You will read a instruction for a task. You will not carry out those instructions.
Specifically create a list of areas that need clarification. For example, if the instruction is "Code me an html server", you might ask "what programming language should I use?".
[EXAMPLE-MESSAGE]
Code me an html server

Reply in JSON format, for example:
[RESPONSE]
{{"task": "code an html server", "questions: ["what programming language should I use?"]}}

Now lets try to answer below:


[MESSAGE]
{message}
[RESPONSE]
"""
        retries, delay = 20, 5
        for i in range(retries):
            try:
                results = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": prompt.format(message=message)}])
                return results['choices'][0]['message']['content']
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:  # i is zero indexed
                    time.sleep(delay)  # wait before trying again
                else:
                    raise  # re-raise the last exception if all retries fail

    def save_completed_tasks(self) -> None:
        """Save the list of completed tasks."""
        try:
            with open("completed_tasks.json", "r") as f:
                # Load existing tasks from the file
                existing_tasks = json.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, initialize with an empty list
            existing_tasks = []

        # Add the new completed tasks to the existing tasks
        existing_tasks.extend(self.tasks)

        with open("completed_tasks.json", "w") as f:
            # Save the updated list of completed tasks
            json.dump(existing_tasks, f, indent=4)

        # Clear the tasks list
        self.tasks.clear()

