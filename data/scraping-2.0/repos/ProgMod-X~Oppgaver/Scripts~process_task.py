import openai
import json
import os
from nbformat import v4 as nb

# OpenAI API Key
openai.api_key = "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"

def generate_solution(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def process_task(task):
    with open(task["README"], "r") as oppgave_file:
        oppgave_text = oppgave_file.read()

    with open(task["solve"], "r") as solve_file:
        solve_code = solve_file.read()

    prompt = f"# Oppgave\n\n{oppgave_text}\n\n# Løsningsforslag\n\n```python\n{solve_code}\n```"
    solution = generate_solution(prompt)

    notebook = nb.new_notebook()
    notebook.cells.append(nb.new_markdown_cell(prompt))
    notebook.cells.append(nb.new_markdown_cell(solution))

    with open(task["solution"], "w") as notebook_file:
        nb.write(notebook, notebook_file)

def main():
    with open("task_index.json", "r") as index_file:
        task_index = json.load(index_file)

    for task_path, task_files in task_index.items():
        if not os.path.exists(task_files["solution"]):
            process_task(task_files)

if __name__ == "__main__":
    main()
