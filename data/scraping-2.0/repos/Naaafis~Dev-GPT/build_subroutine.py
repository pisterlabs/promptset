from reactManager import ReactAppManager

import openai
import diskcache 

# import subroutine classes
from routines.subroutines.file_find import FileFindRoutine
from routines.subroutines.stub_write import StubWriteRoutine
from routines.subroutines.code_write import CodeWriteRoutine
from routines.subroutines.debug import DebugRoutine
from routines.plan import PlanRoutine

# import subroutine function configs
from config.functions import *

class SubroutineBuilder:
    def __init__(self, api_key, app_name, high_level_task):
        self.app_name = app_name
        self.high_level_task = high_level_task
        self.api_key = api_key
        self.user_prompt = high_level_task
        
        # set up configurations for all none-function executing assitants
        self.config_list = [{'model': 'gpt-4-1106-preview', 'api_key': self.api_key}]
        self.base_config = {
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        self.react_manager = ReactAppManager(self.app_name)

        # Initialize config and function maps for each routine.
        self.init_subroutine_configs()
        self.planner = PlanRoutine(self.base_config, self.plan_config, self.plan_function_map)
        # self.find_files = FileFindRoutine(self.base_config, high_level_task, self.file_contents_config, self.file_writing_config, self.file_creating_config, self.find_files_function_map)
        self.stub_writing = StubWriteRoutine(self.base_config, self.stub_reading_config, self.stub_writing_config, self.stub_writing_function_map)
        self.code_writing = CodeWriteRoutine(self.base_config, self.code_reading_config, self.code_writing_config, self.code_writing_function_map)
        self.debugging = DebugRoutine(self.base_config, self.debugging_reading_config, self.debugging_config, self.debugging_function_map)

        self.perform_subroutines()
        
    def append_files_to_task_description(self, high_level_task, file_names):
        return f"{high_level_task}. Involved files: {file_names}"
    
        '''
        # Example usage: 
        high_level_task = "Create a service worker component"
        file_names = ["src/App.js", "src/components/Component.js", "src/utils/helpers.js"]
        updated_task_description = append_files_to_task_description(high_level_task, file_names)
        '''

    
    def perform_subroutines(self):
        # Perform routines for file finding, stub writing, code writing, and debugging.
        # Each of these routines will correspond to a phase in the development process.
        # These will interact with reactManager to perform tasks.
        
        
        # self.planner.init_plan(self.user_prompt)
        # self.installer.find_dependencies(self.react_manager.read_file("", "plan.txt"))
        plan_items = self.react_manager.get_plan_items()
        if not plan_items:
            print("PLAN ROUTINE")
            self.planner.init_plan(self.user_prompt)
            plan_items = self.react_manager.get_plan_items()

        print("PLAN INITIALIZED")
        print("PLAN ITEMS: ", plan_items)
        
        # #for t in range(len(plan_items)):
        # task_list = plan_items[1]
        # full_task = "\n".join(task_list)
        # for step in range(1, len(task_list)):
        #     self.find_files = FileFindRoutine(self.base_config, task_list[step], self.file_contents_config, self.file_writing_config, self.file_creating_config, self.find_files_function_map)
        #     # create the relecant_files.txt to keep track of the files that are relevant to the high_level_task
        #     success = self.react_manager.create_new_file("", "relevant_files.txt", "")
        #     if not success:
        #         print("Error creating relevant_files.txt")
        #         return
            
        #     print("FIND FILES ROUTINE")
        #     status = self.find_files.find_files()
        #     if not status:
        #         print("Error finding files")
        #         return

        #     # read in files names from relevant_files.txt
        #     file_names_str = self.react_manager.read_file("", "relevant_files.txt")

        #     if not file_names_str:
        #         print("Error reading relevant_files.txt")
        #         return
            
        #     # make sure that file_names_str is a string with comma separated file names
        #     # Check if file_names_str contains multiple file paths
        #     if ',' in file_names_str:
        #         file_names = file_names_str.split(", ")
        #     else:
        #         file_names = [file_names_str]  # Wrap the single file path in a list

        #     updated_task_description = self.append_files_to_task_description(self.high_level_task, file_names_str)
            
        #     for file in file_names:
        #         print("File: ", file)
        #         print("STUB WRITING ROUTINE")
        #         print(self.stub_writing.stub_write(file, updated_task_description))
                
        #         print("CODE WRITING ROUTINE")
        #         print(self.code_writing.code_write(file, updated_task_description))

        # print("LINT")
        # numFailed, output = self.react_manager.lint()
        # while(numFailed):
        #     self.debugging.debug(full_task, output)
        #     numFailed, output = self.react_manager.lint()

        # print("EXEC TEST")
        # numFailed, output = self.react_manager.exec_tests()
        # while(numFailed):
        #     self.debugging.debug(full_task, output)
        #     numFailed, output = self.react_manager.exec_tests()
            
        # print("DONE")
        
    
    def init_subroutine_configs(self):
        # Define the groupchat's configs for planning the development process. 
        self.plan_function_map={
            "read_plan": self.react_manager.read_file,
            "create_plan": self.react_manager.create_new_file,
            "write_to_plan": self.react_manager.write_to_file,
            "insert_into_plan": self.react_manager.insert_into_file,
            "delete_lines": self.react_manager.delete_lines,
        }

        self.plan_config = {
            "functions": plan_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        # Define the groupchat's configs for finding relevant files based on the high_level_task.
        # This will likely involve interaction with the ReactAppManager to read files and create relevant ones.
        self.find_files_function_map = {
            "read_file": self.react_manager.read_file,
            "create_new_file": self.react_manager.create_new_file,
            "write_to_file": self.react_manager.write_to_file,
            "list_react_files": self.react_manager.list_react_files,
        }
        
        self.file_contents_config = {
            "functions": file_contents_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        self.file_creating_config = {
            "functions": file_creating_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        self.file_writing_config = {
            "functions": file_writing_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        # Define the groupchat's configs for writing stubs to the files found in the find_files_routine.
        # This will involve reading file contents and adding 'TODO' comments or function stubs.
        self.stub_writing_function_map = {
            "read_file": self.react_manager.read_file,
            "write_to_file": self.react_manager.write_to_file,
        }
        
        self.stub_writing_config = {
            "functions": stub_writing_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        
        self.stub_reading_config = {
            "functions": stub_reading_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        # Define the groupchat's configs for writing the actual code based on the stubs added in the stub_writing_routine.
        # This routine will turn the stubs into executable code.
        self.code_writing_function_map = {
            "read_file": self.react_manager.read_file,
            "write_to_file": self.react_manager.write_to_file,
        }
        
        self.code_writing_config = {
            "functions": code_writing_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        self.code_reading_config = {
            "functions": code_reading_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        # Define the groupchat's configs for debugging the code written in the code_writing_routine.
        # This will check for errors and ensure the code meets the high_level_task requirements.
        self.debugging_function_map = {
            "read_file": self.react_manager.read_file,
            "write_to_file": self.react_manager.write_to_file,
        }
        
        self.debugging_config = {
            "functions": debugging_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
        
        self.debugging_reading_config = {
            "functions": debug_reading_functions,
            "timeout": 600,
            "seed": 42,
            "config_list": self.config_list,
            "temperature": 0,
        }
            

# The following are stubs and will need to be filled in with the actual logic.
def main():
    # Entry point for the script.
    # Parse arguments and create an instance of SubroutineBuilder.
    # Start the routines for the development process.
    
    # read in the user_prompt and web_elements from the front_end
    
    # Create temporary react manager to read file
    TempReactAppManager = ReactAppManager("front_end")
    user_prompt = TempReactAppManager.read_file("saves", "user_prompt.txt")
    web_elements = TempReactAppManager.read_file("saves", "web_elements.txt")
    
    combined_prompt = user_prompt + web_elements
    
    print("combined_prompt: ", combined_prompt)

    subroutineBuilder = SubroutineBuilder("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", "demo-day-app", combined_prompt)
    subroutineBuilder.perform_subroutines()

    
    # ReactAppManager = ReactAppManager("subroutine-app")
    # list_of_files = ReactAppManager.get_react_app_directory()


if __name__ == "__main__":
    main()


# FUTURE PLAN: CREATE SELENIUM WEBDRIVER TO READ DOCUMENTATIONS WHILE WEB BROWSING AGENT FINDS RELEVANT WEBSITES
# 1 agent to read documentation using selenium webdriver
# 1 agent to find relevant websites using web browsing
# 1 agent to review the documentation and websites found
# 1 agent to modify plan based on the documentation and websites found

# This agent can literally go in any of the routines. It can be used to find documentation for the high_level_task.
# It can be used to install the proper libraries for the high_level_task.
# It can be used to identify proper file structure for the high_level_task.
# It can be used to identify proper code structure for the high_level_task for each of the files.