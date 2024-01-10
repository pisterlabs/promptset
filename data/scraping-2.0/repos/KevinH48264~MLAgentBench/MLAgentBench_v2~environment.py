"""
This file contains the Environment class, which prepares the environment for the research agent to run in.

Requirements:
1. This should have access to the workspace and be clear about what the research problem is.
2. This should have access to the actions.py file.

Add-ons:
- maybe give a list of libraries that are already installed?
"""

import json
import os
import sys
import subprocess
import selectors
import shutil
import copy
import time
import fnmatch
import signal
from traceback import format_exception
from multiprocessing import active_children
import readline # to make sure input() works properly
from dacite import from_dict
import functools
from openai import OpenAI
import openai
from dotenv import load_dotenv
from .LLM import complete_text_fast, complete_text_openai, complete_text
load_dotenv()

import MLAgentBench_v2.high_level_actions as high_level_actions
from .schema import Step, Trace, EnvException, TooLongPromptError, LLMError, EnhancedJSONEncoder 
from .LLM import complete_text_claude
from .prepare_task import prepare_task
from MLAgentBench_v2.actions import TOOL_DESCRIPTIONS
from MLAgentBench.high_level_actions import understand_file, append_to_research_log, inspect_script_lines, edit_script, edit_script_lines, reflection, retrieval_from_research_log
from MLAgentBench.low_level_actions import list_files, read_file, write_file, append_file, copy_file, undo_edit_script, execute_script, python_repl, request_help

class Environment:
    def __init__(self, args):
        # Note: This function should be given to the agent to figure out how to use the environment variables.
        print("Initializing environment...")
        self._args = args # Might be able to be deleted, more for other potentially deletable environment functions to use like signal alarm

        # Set up workspace and research problem.
        with open('MLAgentBench_v2/research_problem.txt', 'r') as f:
            self._research_problem = f.read() # self.R(s) = reward model of current state
        self._benchmark_folder_name = args.task
        self._work_dir = prepare_task(
            work_dir = args.work_dir, 
            task_name = args.task, 
            task_type = args.task_type
        )
        self.files = os.listdir(self.work_dir)
        self.max_states = 8
        self.answer_states = [{
            "action": "None",
            "result": "None",
            "answer_state": "None",
            "files": self.files,
        }] # s_t = [(s_t-5, answer_state, files)..., (s_t-2, answer_state, files), (s_t-1, answer_state, files), (s_t, answer_state, files)]. # potentially, we can add a research_log of steps that were taken to achieve a state and help guide future steps to be taken, like a MCTS

        # Set up actions
        self._tool_descriptions = TOOL_DESCRIPTIONS # Formatted for OpenAI function calling
        self._available_actions = {
                # 'understandFile': understand_file,
                # 'appendSummaryToResearchLog': append_to_research_log,
                # 'inspectScriptLines': inspect_script_lines,
                # 'editScript': edit_script,
                # 'editScriptSegment': edit_script_lines,
                'reflection': self.reflection,
                # 'retrievalFromResearchLog': retrieval_from_research_log,
                # 'listFiles': self.list_files,
                'readFile': self.read_file,
                'writeFile': self.write_file,
                # 'appendFile': append_file,
                # 'copyFile': copy_file,
                # 'undoEditScript': undo_edit_script,
                'executeScript': self.execute_script,
                # 'pythonREPL': python_repl,
                # 'requestHelp': self.request_help,
                # 'finalAnswer': self.final_answer,
                # 'webSearch': self.web_search,
                # 'openaiAssistantCreateAssistant': pass,
                # 'openaiAssistantCreateThread': pass,
                # 'openaiAssistantCreateThreadMessage': pass,
                # 'openaiAssistantCreateRun': pass,
                # 'openaiAssistantListThreadMessageCompletion': pass,
            }
        # self.final_answer = False

        # Assistants API specific instantiation
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.model = args.llm_name

        # Set up logging, overwrite existing logs if they exist
        self._log_dir = args.log_dir
        if os.path.exists(self._log_dir):
            shutil.rmtree(self._log_dir)
        os.makedirs(self._log_dir)
        self.main_log_path = os.path.join(self.log_dir, "main_log.txt")
        with open(self.main_log_path, 'w') as f:
            pass
        self.num_steps = 0
        self._start_time = time.time()

        # Other variables in a partially observable Markov Decision Process
        # self.transition = None # Transition probabilities between states. Problem, how do you operate when you don't even know what s' is until you take action a from state s?
        # self.reward = S x A = reward function. # LLM. The agent is the reward modeler based on the Eureka paper. 

        # Checks
        assert(self.research_problem is not None)
        assert("workspace" in self.work_dir and "branch" in self.work_dir) # we should only list files in the workspace and branch
        assert(len(self.tool_descriptions) == len(self.available_actions.keys())) # action descriptions should be the same as action functions


    ############## for logging ##############

    # Logging decorator
    def log_decorator(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Update files
            self.files = os.listdir(self.work_dir)
            kwargs['work_dir'] = self.work_dir # Update to actual work_dir
            assert(kwargs['work_dir'] == self.work_dir) # Ensure that the work_dir sent into any function is the work directory and nothing else

            # Update research log
            try:
                print(f"\nStep: {self.num_steps}\nCalling function {func.__name__}(args = {args}, kwargs = {kwargs})\n")

                # Log the function call
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'work_dir'}
                # with open(self.main_log_path, "a", 1) as log_file:
                #     log_file.write(f"\nStep: {self.num_steps}\nCalling function {func.__name__}({args}, {filtered_kwargs})\n")

                # Perform the actual function
                result = func(*args, **kwargs)

                print("--- TOOL SUCCESS ---")
            except TooLongPromptError:
                result = "EnvError: too long input for the tool"
                print("--- TOOL ERROR ---", e)
            except LLMError as e:
                result = "LLMError: " + e.message
                print("--- TOOL ERROR ---", e)
            except EnvException as e:
                result = "EnvError: " + e.message
                print("--- TOOL ERROR ---", e)
            except TypeError as e:
                invalid_action_error = f"The arguments needs to have proper entries. You may have missed some entries or used inappropriate ones. Please use the correct format and try again:\n{self.tool_descriptions}"
                result = "EnvError: " + invalid_action_error
                print("--- TOOL ERROR ---", e)
            # except TimeoutException as e:
            #     raise e
            #     print("--- TOOL ERROR ---", e)
            except Exception as e:
                result = f"EnvError: Error executing {func.__name__}."
                print("--- TOOL ERROR ---", e)
            print("Finished!")

            # Log the function output
            # with open(self.main_log_path, "a", 1) as log_file:
            #     log_file.write(f"\nFunction {func.__name__} returned: \n{result}\n")

            # Copy work_dir if it exists
            if self.work_dir and os.path.exists(self.work_dir):
                dest_dir = os.path.join(self.log_dir, f"{self.num_steps}_work_dir")
                shutil.copytree(self.work_dir, dest_dir, dirs_exist_ok=True)

            self.num_steps += 1
            
            # Update states
            kwargs['work_dir'] = "." # replace work_dir for the agent to stay in its workspace
            self.update_states(action=f"Calling function {func.__name__}(args = {args}, kwargs = {kwargs})", result=result)
            
            # Log most recent state
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nStep: {self.num_steps}\n{json.dumps(self.answer_states[-1], indent=4)}\n")

            return result
        return wrapper
    
    def update_states(self, action, result):
        """Update the states of the agent based on action and result. 
        TODO: Extra 1: Break up the state into 1) problem 2) current best answer 3) metric 4) problem to solve 5) next step / plan to solve the problem -- some kind of structure like that.

        TODO: If you don't use Assistants API, then you can have one action at a time and then the update state should only use the current action, result, and state to be the new state instead of the entire history. 2) Then you should have a MCTS to plan what is the next move. Or the updated state should just say what is missing, and not say how to fix it.
        """

        # Update files
        self.files = os.listdir(self.work_dir)

        system_prompt = '''You are a helpful assistant. Given a research problem, your goal is to improve the answer.
        
        You will be given this information:
        Research Problem: ...
        Current Files: ...
        Tools / functions: ...
        Most recent files, action, result, and answer states (oldest to newest): ...

        You should then respond to me with your best answer given the new action and result taken, problems that still exist if you haven't solved the research problem, and a plan to solve those problems.
        '''

        user_prompt = f'''Research Problem: {self.research_problem}
        Current Files: {self.files}
        Tools / functions: {self.available_actions.keys()}
        Most recent files, action, result, and answer states (oldest to newest): {self.answer_states}  
'''
        new_answer_state = complete_text_openai(prompt=user_prompt, system_prompt=system_prompt, model=self.model, log_file=self.main_log_path)

        self.answer_states.append({
            "action": action,
            "result": result,
            "answer_state": new_answer_state,
            "files": self.files,
        })
        while len(self.answer_states) > self.max_states:
            self.answer_states.pop(0)

    ############## for actions ##############
    
    def reflection(self, **kwargs):
        @self.log_decorator
        def wrapped_reflection(things_to_reflect_on="", work_dir = ".", **kwargs):
            formatted_answer_states = ""
            for idx, answer_state in enumerate(self.answer_states):
                formatted_answer_states += "\nStep: " + str(idx) 
                formatted_answer_states += "\nFiles: " + str(answer_state['files']) 
                formatted_answer_states += "\nAction: " + answer_state['action'] 
                # formatted_answer_states += "\nResult: " + answer_state['result'] 
                formatted_answer_states += "\nAnswer: " + answer_state['answer_state'] 

            prompt = f"""We are trying to solve this research problem: {self.research_problem}

            Your current research log:
            ```
            {formatted_answer_states}
            ```

            Reflect on this: {things_to_reflect_on} 
            
            Give an answer in natural language paragraphs as truthfully as possible. 
            """

            reflection = complete_text(prompt, model=self.model)
            return f"Reflection: {reflection}\n"
        return wrapped_reflection(**kwargs)

    def list_files(self, **kwargs):
        @self.log_decorator
        def wrapped_list_files(**kwargs):
            return list_files(**kwargs)
        return wrapped_list_files(**kwargs)

    def read_file(self, **kwargs):
        @self.log_decorator
        def wrapped_read_file(file_name, work_dir = '.', max_char_read = 5000, **kwargs):
            assert("workspace" in self.work_dir and "branch" in self.work_dir) # we should only list files in the workspace and branch
            try:
                observation = open(os.path.join(work_dir, file_name)).read()
                return observation[:max_char_read]
            except:
                raise EnvException(f"cannot read file {file_name}")
        return wrapped_read_file(max_char_read = 2000, **kwargs)

    def write_file(self, **kwargs):
        @self.log_decorator
        def wrapped_write_file(file_name='', content='', **kwargs):
            try:
                # Extract the directory path from the full file path and create directory if necessary
                directory = os.path.dirname(os.path.join(self.work_dir, file_name))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Write the file
                with open(os.path.join(self.work_dir, file_name), "w") as f:
                    f.write(content)
                observation = f"File {file_name} written successfully."
                return observation
            except:
                raise EnvException(f"cannot write file {file_name}")
        return wrapped_write_file(**kwargs)

    # TODO: add the "check_file_in_work_dir" function from before
    def execute_script(self, **kwargs):
        @self.log_decorator
        def wrapped_execute_script(script_name, work_dir = ".", **kwargs):
            assert("workspace" in self.work_dir and "branch" in self.work_dir) # we should only list files in the workspace and branch

            if not os.path.exists(os.path.join(work_dir, script_name)):
                raise EnvException(f"The file {script_name} does not exist.")
            try:
                script_path = script_name
                device = kwargs.get("device", "0")  # Default device is "0"
                python = kwargs.get("python", "python")  # Default Python command is "python"

                cmd = f"CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir) # this sets the path for execution!

                stdout_lines = []
                stderr_lines = []

                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stderr, selectors.EVENT_READ)

                while process.poll() is None and selector.get_map():
                    events = selector.select(timeout=1)

                    for key, _ in events:
                        line = key.fileobj.readline()
                        if key.fileobj == process.stdout:
                            print("STDOUT:", line, end =" ")
                            stdout_lines.append(line)
                        else:
                            print("STDERR:", line, end =" ")
                            stderr_lines.append(line)

                for line in process.stdout:
                    line = line
                    print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                for line in process.stderr:
                    line = line
                    print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

                return_code = process.returncode

                if return_code != 0:
                    observation = "".join(stderr_lines)
                else:
                    observation = "".join(stdout_lines)
                if observation == "" and return_code == 0:
                    # printed to stderr only
                    observation = "".join(stderr_lines)

                return "The script has been executed. Here is the output:\n" + observation + "\nSTDOUT:\n" + "".join(stdout_lines) + "\nSTDERR:\n" + "".join(stderr_lines)
            except Exception as e:
                raise EnvException(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
        return wrapped_execute_script(**kwargs)

    def request_help(self, **kwargs):
        @self.log_decorator
        def wrapped_request_help(**kwargs):
            return request_help(**kwargs)
        return wrapped_request_help(**kwargs)

    def final_answer(self, **kwargs):
        @self.log_decorator
        def wrapped_final_answer(**kwargs):
            self.final_answer = kwargs.get('final_answer', "No final answer was submitted as an argument.")
            return "You have successfully submitted your final answer. No more actions necessary."
        return wrapped_final_answer(**kwargs)
    
    def web_search(self, **kwargs):
        @self.log_decorator
        def wrapped_web_search(query = '', **kwargs):
            try:
                web_search_res = input(f"Query: {query} | Result: ") # temporary quick way for web searching
                return web_search_res
            except:
                raise EnvException(f"Web search failed.")
        return wrapped_web_search(**kwargs)
    

    ############################## internal functions ########################################

    def __enter__(self):
        # set time out
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.args.max_time)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):  
        # save error message
        active = active_children()
        print(f'Active Children: {len(active)}')
        # terminate all active children
        for child in active:
            child.terminate()
        # block until all children have closed
        for child in active:
            child.join()
        # report active children
        active = active_children()
        print(f'Active Children: {len(active)}')
            
        if traceback is not None:
            print("Error message saved in error.txt")
            open(os.path.join(self.log_dir, "error.txt"), "w").write(''.join(format_exception(exc_type, exc_value, traceback)))
        open(os.path.join(self.log_dir, "overall_time.txt"), "w").write(str(time.time() - self.start_time))
           
    
    ############################## getters ########################################

    @property
    def research_problem(self):
        return self._research_problem

    @property
    def benchmark_folder_name(self):
        return self._benchmark_folder_name

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def tool_descriptions(self):
        return self._tool_descriptions

    @property
    def available_actions(self):
        return self._available_actions
    
    @property
    def args(self):
        return self._args

    @property
    def start_time(self):
        return self._start_time
     
    ################################# public functions ########################################

    def is_final(self):
        """Check if the task has reached a final state, either by reaching the maximum steps or time, or because the agent has submitted a final answer. """
        if self.num_steps >= self.args.max_steps or time.time() - self.start_time > self.args.max_time:
            return True, None
            
        if self.final_answer:
            final_answer_evaluation = input(f"\nFinal answer submitted: {self.final_answer} Did the agent submit a valid final answer? If yes, respond with 'yes'. If not, provide feedback. ")
            if final_answer_evaluation == "yes":
                return True, None
            return False, final_answer_evaluation
        
        return False, None