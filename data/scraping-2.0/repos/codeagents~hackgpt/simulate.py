import os
import openai
import json
from enum import Enum
from typing import List, Tuple

openai.api_key = os.getenv("OPENAI_API_KEY")


class Agent:
    def __init__(self, name: str):
        self.name = name


class ActionType(Enum):
    COMMUNICATE = "communicate"
    CODE = "code"
    REVIEW = "review"
    GENERATE_TASK = "generate_task"


class Task:
    def __init__(self, agent_name: str, action: ActionType, content: str):
        self.agent_name = agent_name
        self.action = action
        self.content = content

    def communicate(self, other_agent: Agent) -> str:
        print(f"{self.agent_name} -> {other_agent.name}: {self.content}")
        return f"{self.agent_name} -> {other_agent.name}: {self.content}"

    def code(self, objective: str, code_history) -> str:
        code_history_str = '\n####\n'.join(str(element) for element in code_history)
        prompt = f"Goal: {objective}.\n" \
f"Previous code history:\n```\n{code_history_str}\n```\n" \
f"Write code to achieve this. Output the code wrapped in a valid JSON string. Only output JSON and nothing else."
        code = get_prediction(prompt)
        print(f"{self.agent_name} wrote the following code:\n{code}")
        return code

    def review(self, code: str) -> str:
        prompt = f"Review the following code and provide suggestions for improvements in JSON format:\n{code}"
        review = get_prediction(prompt)
        print(f"{self.agent_name} reviewed the code and provided the following suggestions in JSON format:\n{review}")
        return f"{self.agent_name} reviewed the code and provided the following suggestions in JSON format:\n{review}"

    def generate_task(self, objective: str, other_agent: Agent):
        prompt = f"Break down the following objective into smaller, actionable tasks and return the tasks as a JSON list." \
f"Each task should be a valid JSON as per fields in this class:" \
f"""
class Task:
    def __init__(self, agent_name: str, action: ActionType, content: str):
        self.agent_name = agent_name
        self.action = action
        self.content = content

- Agent name can be "Agent1" or "Agent2".
- Action can be "communicate", "code" or "review".
- Content is a natural language description of the task.
""" \
f"{objective}"
        tasks_json = get_prediction(prompt)
        tasks_list = json.loads(tasks_json)

        new_tasks = []
        for task_desc in tasks_list:
            task = Task(task_desc['agent_name'], ActionType(task_desc['action']), task_desc['content'])
            new_tasks.append(task)

        print(new_tasks)
        return new_tasks


def get_prediction(prompt):
    print(f"## Received prompt: {prompt}")
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    print(f"## Response: {completion.choices[0].message.content}")
    return completion.choices[0].message.content


def parse_objective(objective: str) -> list[str]:
    """
    Extracts code from a string delimited by triple tick markers (```).

    Args:
        objective (str): the string containing natural language and code.

    Returns:
        A list of strings representing the code blocks found in the input string.
    """
    code_blocks = []
    start_idx = 0

    while True:
        start_idx = objective.find("```", start_idx)
        if start_idx == -1:
            break

        end_idx = objective.find("```", start_idx + 3)
        if end_idx == -1:
            break

        code_blocks.append(objective[start_idx + 3: end_idx].strip())
        start_idx = end_idx + 3

    return code_blocks


def main():
    objective = open('objective.txt', 'r').read()

    agent1 = Agent("Agent1")
    agent2 = Agent("Agent2")

    task_queue: List[Task] = []

    # Add tasks to the task queue
    task_queue.append(Task(agent1.name, ActionType.COMMUNICATE, f"Let's work on the objective:\n{objective}"))
    task_queue.append(Task(agent2.name, ActionType.COMMUNICATE, "I agree, let's break it down into smaller tasks!"))
    task_queue.append(Task(agent1.name, ActionType.GENERATE_TASK, ""))

    # Process the task queue and collect results
    results = []
    code_history = []
    code_snippets_in_objective = parse_objective(objective)
    for snippet in code_snippets_in_objective:
        code_history.append(snippet)
    # Process the new tasks generated
    code_results = []
    idx = 0
    while idx < len(task_queue):
        task = task_queue[idx]
        print(f'Running task {idx}.')
        if task.action == ActionType.CODE:
            code_result_json = task.code(task.content, objective if not code_history else code_history).strip('```')
            code_result_json = code_result_json.strip()
            code_result = str(json.loads(code_result_json).get('code'))
            code_results.append(code_result)
            code_history.append(code_result)
        elif task.action == ActionType.REVIEW:
            code_to_review = "\n=====\n".join(str(x) for x in code_history)  # Provide all the generated code so far
            review_result = task.review(code_to_review)
            results.append(review_result)
        elif task.action == ActionType.COMMUNICATE:
            if task.agent_name == agent1.name:
                results.append(task.communicate(agent2))
            else:
                results.append(task.communicate(agent1))
        elif task.action == ActionType.GENERATE_TASK:
            new_tasks = task.generate_task(objective, agent2)
            task_queue.extend(new_tasks)
            results.append(f"{task.agent_name} generated the following tasks:\n" + "\n".join(t.content for t in new_tasks))
        idx += 1

    return tuple(results), code_results


if __name__ == "__main__":
    (communication1, communication2, generated_tasks, review), code_results = main()
    print(communication1)
    print(communication2)
    print(generated_tasks)
    for _code_result in code_results:
        print(_code_result)
    print(review)
