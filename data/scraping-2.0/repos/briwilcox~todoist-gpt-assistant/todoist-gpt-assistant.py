import os
import datetime
import configparser
import openai
from todoist_api_python.api import TodoistAPI
from colorama import Fore, Style, init
import argparse
from tqdm import tqdm


# Initialize colorama
init()
# Read configuration
config = configparser.ConfigParser()
config.read("todoist-config.ini")
todoist_api_key = config['todoist']['api_key']
openai_api_key = config['openai']['api_key']

# Get the seven most recent tasks that are due today or earlier
def get_seven_most_recent_tasks(api, offset=0):
    tasks = api.get_tasks()
    tasks.sort(key=lambda task: task.due.date if task.due else '', reverse=True)
    tasks_due_today_or_earlier = [
        task for task in tasks
        if task.due and datetime.datetime.strptime(task.due.date, '%Y-%m-%d').date() <= datetime.date.today()
        and "every" not in task.due.string.lower()
        and "SUGGESTION" not in task.description.upper()
    ]

    return tasks_due_today_or_earlier[offset:offset + 7]

# Update the task description with the suggestion
def update_task_description(api, task, suggestion, update_all, no_update):
    existing_content = task.description

    if no_update:
        return
    
    if "SUGGESTION" in existing_content.upper() and not update_all:
        return

    if existing_content:
        new_description = f"{existing_content}\n\n{suggestion}"
    else:
        new_description = f"{suggestion}"

    try:
        api.update_task(task_id=task.id, description=new_description)
    except Exception as error:
        print(error)



# Generate a suggestion for how to accomplish a task, and select which model to use, as well as the token budget, and temperature
def generate_suggestions(task, model_name="gpt-3.5-turbo", max_token_budget=200, model_temperature=0.7):
    #check if model exists and is gpt-4 or gpt-3.5-turbo
    if model_name not in ["gpt-4", "gpt-3.5-turbo"]:
        model_name = "gpt-3.5-turbo" # default to gpt-3.5-turbo if model name is not gpt-4 or gpt-3.5-turbo
    prompt = f"Please suggest how to accomplish the task in under 600 characters including the task name, do not add empty lines, do not simply repeat the task in the suggestion tell me how to accomplish it with specifics: {task}"
    response = openai.ChatCompletion.create(
        model=model_name,
        messages = [
        {"role": "system", "content": "You are a helpful assistant who provides advice, usually in the form of lists, on how to accomplish a given goal."}, 
        {"role": "user", "content": prompt}
        ],
        max_tokens=max_token_budget,
        n=1,
        stop=None,
        temperature=model_temperature,
    )
    suggestion = response.choices[0].message['content'].strip()
    return f"{model_name} SUGGESTION: {suggestion}"

# Parse command line arguments
def main(interactive, update_all, due_today):
    api = TodoistAPI(todoist_api_key)
    openai.api_key = openai_api_key
    offset = 0

    while True:
        if update_all:
            tasks = api.get_tasks()
            if due_today:
                tasks = [task for task in tasks if task.due and datetime.datetime.strptime(task.due.date, '%Y-%m-%d').date() <= datetime.date.today()]
            tasks_to_update = [task for task in tasks if "SUGGESTION" not in task.description.upper()]
        else:
            tasks_to_update = get_seven_most_recent_tasks(api, offset)

        if not tasks_to_update and update_all:
            print("No tasks to update. This is probably because you have already updated all tasks with suggestions.")
            break

        # Wrap tasks_to_update with tqdm to show progress bar
        for task in tqdm(tasks_to_update, desc="Updating tasks", unit="task"):
            suggestion = generate_suggestions(task.content)

            print(f"{Fore.GREEN}TASK: {task.content}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{suggestion}{Style.RESET_ALL}")
            print()  # Add an empty line for readability
            if update_all:
                update_task_description(api, task, suggestion, update_all, no_update=False)
            elif interactive:
                update_task_description(api, task, suggestion, update_all, no_update=False)
            else:
                update_task_description(api, task, suggestion, update_all, no_update=False)

        if not interactive or update_all:
            break

        while True:
            print("Choose an option:")
            print("1. Get advice on the next 7 tasks. (Type 1, n, or next to continue)")
            print("2. Quit. Type 2, q, or quit to quit.")

            choice = input("Enter the number of your choice: ")

            if choice == "1" or choice.lower() in ["n", "next"]:
                offset += 7
                break
            elif choice == "2" or choice.lower() in ["q", "quit"]:
                print("Goodbye!")
                exit(0)
            else:
                print("Invalid choice. Please try again.")


# Run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate suggestions for Todoist tasks")
    # Add arguments and give option for store_true or store_false
    parser.add_argument("-u", "--update-all", help="Update all task descriptions that haven't been updated yet", action="store_true")
    parser.add_argument("-i", "--interactive", help="Enable interactive mode", action="store_true")
    parser.add_argument("-d", "--due-today", help="Only update tasks that are overdue or due today", action="store_true")
    args = parser.parse_args()
    main(args.interactive, args.update_all, args.due_today)
