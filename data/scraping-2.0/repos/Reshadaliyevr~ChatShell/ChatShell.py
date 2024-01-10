import openai
import re
import subprocess
import sys
import ctypes
from rich.console import Console
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import button_dialog


#run this program as admin when first run 
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
    
if is_admin():
    pass
else:
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

console = Console()


def save_api_key(api_key):
    with open('api_key.txt', 'w') as file:
        file.write(api_key)

def load_api_key():
    try:
        with open('api_key.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


def query_gpt(prompt,api_key):
    #take input as key variable
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text.strip()
# validate if a valid key
def is_valid_key(api_key):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="hello",
            max_tokens = 5
        )
        return True

    except openai.OpenAIError as e:

        return False
def execute_powershell(script):
    process = subprocess.Popen(["powershell.exe", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')

#initializing converstaion history
conversation_history = []

# Keyword that indicate reference to prior conversation
reference_keywords = ["that", "previous", "earlier", "before", "last", "rename"]
def main_menu():
    menu_text = """\nPowerShell Admin Tool
1. Perform a task
2. View task history
3. Clear task history
4. Exit
\nPlease choose an option (1-4): """
    return prompt(menu_text, completer=WordCompleter(["1", "2", "3", "4"]))

def main():
    console.print("Welcome to the PowerShell Admin tool.", style="bold")
    task_history = []
    loaded_api_key = load_api_key()
    if not loaded_api_key or not is_valid_key(loaded_api_key):
        while True:
            api_key = input("Enter your OpenAI API key: ")
            if is_valid_key(api_key):
                save_option = input("Do you want to save the API key for next time? (y/n): ").lower()
                if save_option == 'y':
                    save_api_key(api_key)
                break
    else:
        api_key = loaded_api_key
    while True:
        action = main_menu()
        if action == '2':
            # View task history
            table = Table(title="Task History")
            table.add_column("Index", style="cyan")
            table.add_column("Task", style="magenta")

            for i, past_task in enumerate(task_history, 1):
                table.add_row(str(i), past_task)

            console.print(table)

        elif action == '3':
            # Clear task history
            task_history.clear()
            console.print("[green]Task history cleared.[/green]")

        elif action == '4':
            # Exit
            break

        elif action == '1':
            # Perform a task
            user_task = prompt("Enter a detailed description of the task you want to perform: ")
            
            #Detect if user input contains reference keywords
            use_converstaion_history = any(keyword in user_task.lower() for keyword in reference_keywords)
            #Add user task to converstaion history
            conversation_history.append(f"User: {user_task}")
            
            # Construct the prompt
            prompt_text = "As a senior developer skilled in PowerShell scripting, please provide a PowerShell script for the following task:\n"
            
            # Include conversation history in prompt if flag is set
            if use_converstaion_history:
                prompt_text += "\n".join(conversation_history[-2:]) + "\n\nTask: "

            # Add current task to prompt
            prompt_text += f"{user_task}\n\n```script```\n"
            
            # Query the AI
            response_text = query_gpt(prompt_text, api_key)

            # Add AI response to conversation history
            conversation_history.append(f"AI: {response_text}")

            match = re.search(r'(?s)(?<=```).+?(?=```)|\n\n[^`].*?(?=\n\n|$)', response_text)
            if match:
                powershell_script = match.group(0).strip()
            else:
                powershell_script = response_text.strip()

            console.print(f"\n[bold green]Generated PowerShell script:[/bold green]\n\n{powershell_script}")



            run_script = input("\nDo you want to run this script? (y/n): ")
            if run_script.lower() == 'y':
                output, error = execute_powershell(powershell_script)
                if output:
                    console.print(f"\n[bold green]Output:[/bold green]\n\n{output}")
                if error:
                    console.print(f"\n[bold red]Error:[/bold red]\n\n{error}")
                    # Ask the AI to fix the error
                    error_prompt = f"The PowerShell script generated an error:\n\n{error}\n\nCan you provide a fixed version of the script?\n\nScript:\n\n```{powershell_script}```\n\n"
                    response_text = query_gpt(error_prompt, api_key)
                    match = re.search(r'(?s)(?<=```).+?(?=```)|\n\n[^`].*?(?=\n\n|$)', response_text)
                    if match:
                        fixed_script = match.group(0).strip()
                    else:
                        fixed_script = response_text.strip()

                    console.print(f"\n[bold green]Fixed PowerShell script:[/bold green]\n\n{fixed_script}")

                    # Run the fixed script
                    run_fixed_script = input("\nDo you want to run the fixed script? (y/n): ")
                    if run_fixed_script.lower() == 'y':
                        output, error = execute_powershell(fixed_script)
                        if output:
                            console.print(f"\n[bold green]Output:[/bold green]\n\n{output}")
                        if error:
                            console.print(f"\n[bold red]Error:[/bold red]\n\n{error}")
                    else:
                        console.print("Skipping fixed script execution.")

            else:
                change_script = input("\nDo you want to change the script? (y/n): ")
                if change_script.lower() == 'y':
                    # Ask the AI to fix the script
                    change_prompt = f"I am not satisfied with the following PowerShell script:\n\n```{powershell_script}```\n\nCan you provide an alternative version of the script in this form?: ```script```\n\n"
                    response_text = query_gpt(change_prompt, api_key)
                    match = re.search(r'(?s)(?<=```).+?(?=```)|\n\n[^`].*?(?=\n\n|$)', response_text)
                    if match:
                        alternative_script = match.group(0).strip()
                    else:
                        print("The AI didn't return an alternative script in the expected format. Please try again.")
                        return

                    console.print(f"\n[bold green]Alternative PowerShell script:[/bold green]\n\n{alternative_script}")

                    # Run the fixed script
                    run_alternative_script = input("\nDo you want to run the alternative script? (y/n): ")
                    if run_alternative_script.lower() == 'y':
                        output, error = execute_powershell(alternative_script)
                        if output:
                            console.print(f"\n[bold green]Output:[/bold green]\n\n{output}")
                        if error:
                            console.print(f"\n[bold red]Error:[/bold red]\n\n{error}")
                    else:
                        console.print("Skipping fixed script execution.")
                else:
                    console.print("Skipping fixed script execution.")
            task_history.append(user_task)

if __name__ == "__main__":
    main()
