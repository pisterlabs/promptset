from langchain import LLMChain, OpenAI, PromptTemplate
import json


class TasksModel:
    # temporary: serialize tasks to file for debugging
    def __init__(self, tasks_path="./agent/models/tasks.json"):
        self.tasks_path = tasks_path

    def get_tasks(self, name):
        print(f"\nGetting tasks for {name}...")
        with open(self.tasks_path, "r") as f:
            tasks_json = f.read()

        # Parse the JSON string
        tasks_data = json.loads(tasks_json)

        # Normalize name before lookup
        name = name.lower()

        # Return the tasks for the given name if it exists
        if name in tasks_data["tasks"]:
            return tasks_data["tasks"][name]

        # If the name is not found, print a message and return None
        print(f"No tasks found for {name}")
        return None


if __name__ == "__main__":
    tasks_model = TasksModel()
    tasks_model.get_tasks("janita")
