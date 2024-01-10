import json
import os

import pprint



class Planner():
    def check_plan(self, agent , task):
        """this function checks if the file plan.md exists, if it doesn't exist it gets created"""

        file_name = "plan.md"

        if not agent.workspace.exists(task.task_id,file_name):
            data = """
                    # Task List and status:
                    - [ ] Create a detailed checklist for the current plan and goals
                    - [ ] Finally, review that every new task is completed
                    """
                    # ## Notes:
                    # - Use the run_planning_cycle command frequently to keep this plan up to date.
                    #         """
            if isinstance(data, str):
                data = data.encode()
            agent.workspace.write(task_id=task.task_id, path=file_name, data=data)
            print(f"{file_name} created.")

        return agent.workspace.read(task_id=task.task_id, path=file_name).decode()


    async def update_plan(self, agent , task , chat_history = []):
        """this function checks if the file plan.md exists, if it doesn't exist it gets created"""


        file_name = "plan.md"

        data = agent.workspace.read(task_id=task.task_id, path=file_name).decode()

        response = await self.generate_improved_plan(agent , task , data ,chat_history)

        data = response
        if isinstance(data, str):
            data = data.encode()
        agent.workspace.write(task_id=task.task_id, path=file_name, data=data)

        print(f"{file_name} updated.")

        return response


    async def generate_improved_plan(self, agent , task , prompt: str , chat_history = []) -> str:
        """Generate an improved plan using OpenAI's ChatCompletion functionality"""

        # import openai

        # tasks = self.load_tasks(agent , task)
        tasks = task.input

        model = os.getenv('PLANNER_MODEL', os.getenv('FAST_LLM_MODEL', 'gpt-3.5-turbo'))
        max_tokens = os.getenv('PLANNER_TOKEN_LIMIT', os.getenv('FAST_TOKEN_LIMIT', 1500))
        temperature = os.getenv('PLANNER_TEMPERATURE', os.getenv('TEMPERATURE', 0.5))

        # Call the OpenAI API for chat completion
        messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that improves and adds crucial points to plans in .md format.",
                },
                {
                    "role": "user",
                    "content": f"Create detailed plan include tasks lists to to fulfil this goal:\n{tasks}\n, keep the .md format example:\n{prompt}\n"
                            f"avaliable abilities you can use to fulfil tasks:\n{agent.abilities.list_non_planning_abilities_name_description()}\n",
                },
            ]
        for msg in chat_history:
            messages.append(msg)
        chat_completion_kwargs = {
                "messages": messages,
                "model": model,
                # "max_tokens":int(max_tokens),
                # "n":1,
                # "temperature":float(temperature),
            }
            # Make the chat completion request and parse the response
        print(pprint.pformat(chat_completion_kwargs))
        from .. import chat_completion_request
        response = await chat_completion_request(**chat_completion_kwargs)
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are an assistant that improves and adds crucial points to plans in .md format.",
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Update the following plan given the task status below, keep the .md format:\n{prompt}\n"
        #                     f"Include the current tasks in the improved plan, keep mind of their status and track them "
        #                     f"with a checklist:\n{tasks}\n Revised version should comply with the contents of the "
        #                     f"tasks at hand:",
        #         },
        #     ],
        #     max_tokens=int(max_tokens),
        #     n=1,
        #     temperature=float(temperature),
        # )
        
        

        # Extract the improved plan from the response
        improved_plan = response.choices[0].message.content.strip()
        return improved_plan


    # def create_task(self, agent , task=None, task_description: str = None, status=False):
    #     taskj = {"description": task_description, "completed": status}
    #     tasks = self.load_tasks(agent , task.task_id)
    #     tasks[str(task.task_id)] = taskj

    #     # current_working_directory = os.getcwd()
    #     # workdir = os.path.join(
    #     #     current_working_directory, "auto_gpt_workspace", "tasks.json"
    #     # )
    #     file_name = "tasks.json"

    #     # with open(file_name, "w") as f:
    #     #     json.dump(tasks, f)
    #     data = json.dumps(tasks)
    #     if isinstance(data, str):
    #         data = data.encode()
    #     agent.workspace.write(task_id=task.task_id, path=file_name, data=data)

    #     return tasks


    # def load_tasks(self, agent , task) -> dict:
    #       task_id = task.task_id
    #     # current_working_directory = os.getcwd()
    #     # workdir = os.path.join(
    #     #     current_working_directory, "auto_gpt_workspace", "tasks.json"
    #     # )
    #     file_name = "tasks.json"

    #     # if not os.path.exists(file_name):
    #     #     with open(file_name, "w") as f:
    #     #         f.write("{}")

    #     if not agent.workspace.exists(task.task_id,file_name):
    #         data = "\{\}"
    #         if isinstance(data, str):
    #             data = data.encode()
    #         agent.workspace.write(task_id=task.task_id, path=file_name, data=data)
    #         print(f"{file_name} created.")

    #     try:
    #         tasks = json.loads(agent.workspace.read(task_id=task_id, path=file_name).decode())
    #         if isinstance(tasks, list):
    #             tasks = {}
    #     except json.JSONDecodeError:
    #         tasks = {}


    #     # with open(file_name) as f:
    #     #     try:
    #     #         tasks = json.load(f)
    #     #         if isinstance(tasks, list):
    #     #             tasks = {}
    #     #     except json.JSONDecodeError:
    #     #         tasks = {}

    #     return tasks


    # def update_task_status(self, agent , task):
    #     task_id = task.task_id
    #     tasks = self.load_tasks(agent , task_id)

    #     if str(task_id) not in tasks:
    #         print(f"Task with ID {task_id} not found.")
    #         return

    #     tasks[str(task_id)]["completed"] = True

    #     # current_working_directory = os.getcwd()
    #     # workdir = os.path.join(
    #     #     current_working_directory, "auto_gpt_workspace", "tasks.json"
    #     # )
    #     file_name = "tasks.json"
    #     data = json.dumps(tasks)
    #     if isinstance(data, str):
    #         data = data.encode()
    #     agent.workspace.write(task_id=task_id, path=file_name, data=data)
    #     # with open(file_name, "w") as f:
    #     #     json.dump(tasks, f)

    #     return f"Task with ID {task_id} has been marked as completed."