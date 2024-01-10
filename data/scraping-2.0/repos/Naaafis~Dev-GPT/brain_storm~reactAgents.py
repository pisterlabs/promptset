from reactManager import ReactAppManager
import openai

class GPTAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_code(self, prompt, max_tokens=300):  # Increased token limit
        response = openai.Completion.create(
            engine="gpt-4.0-turbo",
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.choices[0].text.strip()

class TaskCreator:
    def __init__(self, react_manager):
        self.react_manager = react_manager

    def initial_tasks(self):
        tasks = []
        if not self.react_manager.check_react_installed():
            tasks.append("Install React")
        tasks.append("Setup React App")
        return tasks

class InterpreterAgent:
    def __init__(self, gpt_agent):
        self.gpt_agent = gpt_agent

    def interpret(self, user_prompt):
        prompt = f"Break down the user prompt into tasks: {user_prompt}"
        tasks = self.gpt_agent.generate_code(prompt)
        return tasks.split("\n")

class PlanningAgent:
    def __init__(self, gpt_agent, react_manager):
        self.gpt_agent = gpt_agent
        self.react_manager = react_manager

    def process_requirements(self, requirements):
        # List of all available functions in ReactAppManager
        functions = [
            "set_react_app_name", "get_react_app_directory", "get_react_app_name", "get_root_directory",
            "list_react_files", "list_react_directory_contents", "check_os", "check_node_version", 
            "install_node_based_on_os", "check_for_common_package_installers", "check_react_installed", 
            "install_npm_packages", "create_react_app", "npm_start", "stop_react_app", "execute_command", 
            "create_directory", "read_react_file", "create_new_file", "edit_file", "edit_json_file"
        ]

        # Constructing the prompt
        function_string = ", ".join(functions)
        prompt = (f"Given the requirements '{requirements}' and the available ReactAppManager functions: "
                  f"[{function_string}], provide a sequence of tasks that address the requirements.")

        # Generate the plan
        plan = self.gpt_agent.generate_code(prompt)
        return plan.split("\n")


class CodeWritingAgent:
    def __init__(self, gpt_agent):
        self.gpt_agent = gpt_agent

    def write_code(self, task):
        prompt = f"Generate code for the task: {task}"
        code = self.gpt_agent.generate_code(prompt)
        return code

class ExecutorAgent:
    def __init__(self, react_manager):
        self.react_manager = react_manager

    def execute_task(self, task, **kwargs):
        try:
            if task == "set_react_app_name":
                return self.react_manager.set_react_app_name(kwargs.get('app_name'))

            elif task == "get_react_app_directory":
                return self.react_manager.get_react_app_directory()

            elif task == "get_react_app_name":
                return self.react_manager.get_react_app_name()

            elif task == "get_root_directory":
                return self.react_manager.get_root_directory()

            elif task == "list_react_files":
                return self.react_manager.list_react_files()

            elif task == "list_react_directory_contents":
                return self.react_manager.list_react_directory_contents()

            elif task == "check_os":
                return self.react_manager.check_os()

            elif task == "check_node_version":
                return self.react_manager.check_node_version()

            elif task == "install_node_based_on_os":
                return self.react_manager.install_node_based_on_os()

            elif task == "check_for_common_package_installers":
                return self.react_manager.check_for_common_package_installers()

            elif task == "check_react_installed":
                return self.react_manager.check_react_installed()

            elif task == "install_npm_packages":
                packages = kwargs.get('packages', [])
                return self.react_manager.install_npm_packages(packages)

            elif task == "create_react_app":
                app_name = kwargs.get('app_name')
                return self.react_manager.create_react_app(app_name)

            elif task == "npm_start":
                return self.react_manager.npm_start()

            elif task == "stop_react_app":
                return self.react_manager.stop_react_app()

            elif task == "execute_command":
                command = kwargs.get('command')
                return self.react_manager.execute_command(command)

            elif task == "create_directory":
                dir_name = kwargs.get('dir_name')
                return self.react_manager.create_directory(dir_name)

            elif task == "read_react_file":
                filename = kwargs.get('filename')
                return self.react_manager.read_react_file(filename)

            elif task == "create_new_file":
                filename = kwargs.get('filename')
                content = kwargs.get('content', "")
                directory = kwargs.get('directory', None)
                return self.react_manager.create_new_file(filename, content, directory)

            elif task == "edit_file":
                filename = kwargs.get('filename')
                content = kwargs.get('content')
                mode = kwargs.get('mode', 'replace')
                line_num = kwargs.get('line_num', None)
                return self.react_manager.edit_file(filename, content, mode, line_num)

            elif task == "edit_json_file":
                filename = kwargs.get('filename')
                content_str = kwargs.get('content_str')
                return self.react_manager.edit_json_file(filename, content_str)

            else:
                return "Task not recognized."
        
        except KeyError as e:
            return f"Required argument missing: {str(e)}"
        except Exception as e:
            return f"Error executing task {task}: {str(e)}"


class DebuggerAgent:
    def __init__(self, gpt_agent, planner):
        self.gpt_agent = gpt_agent
        self.planner = planner

    def debug(self, error):
        prompt = f"Debug the error and provide a solution that can be translated to tasks: {error}"
        solution = self.gpt_agent.generate_code(prompt)
        
        # Convert the solution into tasks using the planner
        tasks = self.planner.process_requirements(solution)
        return tasks
    
class ComponentPlanningAgent:
    def __init__(self, gpt_agent, executor, react_manager):
        self.gpt_agent = gpt_agent
        self.executor = executor
        self.react_manager = react_manager

    def determine_file_path(self, code):
        # Getting the current structure of the React app
        current_structure = self.get_current_state()
        
        # Constructing the prompt for the GPT Agent to determine the best file path
        prompt = (f"Given the current React app structure:\n\n"
                  f"{current_structure}\n\n"
                  f"And the following code to be written:\n\n"
                  f"{code}\n\n"
                  f"Determine the best file or directory to place the new code. If a new file is needed, specify its name and path.")
        
        # Get the file path decision from the model
        file_path = self.gpt_agent.generate_code(prompt)
        return file_path

    def process_code(self, code):
        # Determine the best file or directory for the code
        file_path = self.determine_file_path(code)
        
        # Constructing the prompt for the GPT Agent to determine line numbers and actions
        prompt = (f"Given the current React app structure and the following code to be written:\n\n"
                  f"{code}\n\n"
                  f"Determine the line numbers and actions (e.g., insert, replace, delete) to be taken "
                  f"on the file: {file_path}. Also, ensure that any necessary dependencies are added.")
        
        # Generate the plan based on the code and the file_path
        plan = self.gpt_agent.generate_code(prompt)
        
        # Parse the plan to extract actions, line numbers, and any additional tasks
        actions = plan.split("\n")
        
        # Execute the actions using the executor
        for action in actions:
            task, *args = action.split(", ")
            if task == "edit_file":
                filename, content, mode, line_num = args
                self.executor.execute_task(task, filename=filename, content=content, mode=mode, line_num=int(line_num))
            elif task == "edit_json_file":
                filename, content_str = args
                self.executor.execute_task(task, filename=filename, content_str=content_str)
        return "Code processed successfully."

class ComponentMaker:
    def __init__(self, api_key, app_name, user_prompt):
        self.react_manager = ReactAppManager(app_name)
        self.gpt_agent = GPTAgent(api_key)
        self.task_creator = TaskCreator(self.react_manager)
        self.interpreter = InterpreterAgent(self.gpt_agent)
        self.planner = PlanningAgent(self.gpt_agent, self.react_manager)
        self.writer = CodeWritingAgent(self.gpt_agent)
        self.executor = ExecutorAgent(self.react_manager)
        self.debugger = DebuggerAgent(self.gpt_agent, self.planner)
        self.component_planner = ComponentPlanningAgent(self.gpt_agent, self.executor, self.react_manager)

        # Initial tasks from TaskCreator
        self.tasks = self.task_creator.initial_tasks()
        
        # Adding tasks from Interpreter
        self.tasks.extend(self.interpreter.interpret(user_prompt))

    def process(self):
        while self.tasks:
            task = self.tasks.pop(0)
            
            # Conversation with CodeWritingAgent
            code = self.writer.write_code(task) # maybe the tasks arent even to write code anymore
            # the coding agent needs to also be aware of what line numbers to write to
            # additionally, the coding agent also needs to know how to read the file and figure out if code even needs to be written
            # these tasks could be done by another "PlanningAgent" that is aware of the current state of the app
            # this plannign agents job is to promt the executor to look for certain things in the app and figure out where the code 
            # needs to be written. Then the coding agent can write the code and return to this planning agent to figure out what to do next
            # this planning agent concludes by adding the contents of the coding agents along with the line numbers to the tasks list
            # the executor agent then executes the task and returns the result to the planning agent
                
            # Conversation with ComponentPlanningAgent
            result = self.component_planner.process_code(code)
                
            # Check for errors in the result
            if "error" in result.lower():
                # Conversation with DebuggerAgent
                new_tasks = self.debugger.debug(result)
                # Add the new tasks to the beginning of the tasks list
                self.tasks = new_tasks + self.tasks

def main():
    maker = ComponentMaker("YOUR_OPENAI_API_KEY", "my_app", "Create a user profile component.")
    maker.process()

if __name__ == "__main__":
    main()

