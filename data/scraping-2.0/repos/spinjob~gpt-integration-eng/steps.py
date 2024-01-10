import inspect
import re
import subprocess
import logging
import os

from enum import Enum
from typing import List, Union

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from termcolor import colored

from gpt_engineer.ai import AI
from gpt_engineer.chat_to_files import to_files
from gpt_engineer.db import DBs
from gpt_engineer.learning import human_input


Message = Union[AIMessage, HumanMessage, SystemMessage]

logging.basicConfig(level=logging.INFO)

MAX_ITERATIONS = 3

def setup_sys_prompt(dbs: DBs) -> str:
    return (
        dbs.preprompts["generate"] + "\nUseful to know:\n" + dbs.preprompts["philosophy"]
    )


def get_prompt(dbs: DBs, project_prompt: str = None) -> str:
    """While we migrate we have this fallback getter"""

    if project_prompt is not None:
        print("Using project_prompt")
        print(project_prompt)
        return project_prompt;
    
    print("Using dbs.input")
    print(dbs.input)

    assert (
        "prompt" in dbs.input or "main_prompt" in dbs.input
    ), "Please put your prompt in the file `prompt` in the project directory"

    if "prompt" not in dbs.input:
        print(
            colored("Please put the prompt in the file `prompt`, not `main_prompt", "red")
        )
        print()
        return dbs.input["main_prompt"]

    return dbs.input["prompt"]


def curr_fn() -> str:
    """Get the name of the current function"""
    return inspect.stack()[1].function


# All steps below have the signature Step


def simple_gen(ai: AI, dbs: DBs, project_prompt: str = None) -> List[Message]:

    print("Running simple_gen")
    print("project_prompt", project_prompt)

    """Run the AI on the main prompt and save the results"""
    messages = ai.start(setup_sys_prompt(dbs), get_prompt(dbs, project_prompt), step_name=curr_fn())
    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def clarify(ai: AI, dbs: DBs, project_prompt: str = None) -> List[Message]:
    print("Running clarify")
    print("project_prompt", project_prompt)
    """
    Ask the user if they want to clarify anything and save the results to the workspace
    """
    messages: List[Message] = [ai.fsystem(dbs.preprompts["clarify"])]
    print("(letting gpt-engineer make its own assumptions)")
    messages = ai.next(
        messages,
        "Make your own assumptions and state them explicitly before starting",
        step_name=curr_fn(),
    )
    # while True:
    #     messages = ai.next(messages, user_input, step_name=curr_fn())
    #     msg = messages[-1].content.strip()

    #     if msg == "Nothing more to clarify.":
    #         break

    #     if msg.lower().startswith("no"):
    #         print("Nothing more to clarify.")
    #         break

    #     print()
    #     user_input = input('(answer in text, or "c" to move on)\n')
    #     print()

    #     if not user_input or user_input == "c":
    #         print("(letting gpt-engineer make its own assumptions)")
    #         print()
    #         messages = ai.next(
    #             messages,
    #             "Make your own assumptions and state them explicitly before starting",
    #             step_name=curr_fn(),
    #         )
    #         print()
    #         return messages

    #     user_input += (
    #         "\n\n"
    #         "Is anything else unclear? If yes, only answer in the form:\n"
    #         "{remaining unclear areas} remaining questions.\n"
    #         "{Next question}\n"
    #         'If everything is sufficiently clear, only answer "Nothing more to clarify.".'
    #     )

    # print()
    return messages


def gen_spec(ai: AI, dbs: DBs, project_prompt: str = None) -> List[Message]:
    print("Running gen_spec")
    """
    Generate a spec from the main prompt + clarifications and save the results to
    the workspace
    """
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fsystem(f"Instructions: {project_prompt}"),
    ]

    messages = ai.next(messages, dbs.preprompts["spec"], step_name=curr_fn())

    dbs.memory["specification"] = messages[-1].content.strip()

    # Original Implementation
        # print("Running gen_spec")
        # print("project_prompt", dbs.input['prompt'])
        # """
        # Generate a spec from the main prompt + clarifications and save the results to
        # the workspace
        # """
        # messages = [
        #     ai.fsystem(setup_sys_prompt(dbs)),
        #     ai.fsystem(f"Instructions: {dbs.input['prompt']}"),
        # ]

        # messages = ai.next(messages, dbs.preprompts["spec"], step_name=curr_fn())

        # dbs.memory["specification"] = messages[-1].content.strip()

    return messages


def respec(ai: AI, dbs: DBs) -> List[Message]:
    messages = AI.deserialize_messages(dbs.logs[gen_spec.__name__])
    messages += [ai.fsystem(dbs.preprompts["respec"])]

    messages = ai.next(messages, step_name=curr_fn())
    messages = ai.next(
        messages,
        (
            "Based on the conversation so far, please reiterate the specification for "
            "the program. "
            "If there are things that can be improved, please incorporate the "
            "improvements. "
            "If you are satisfied with the specification, just write out the "
            "specification word by word again."
        ),
        step_name=curr_fn(),
    )

    dbs.memory["specification"] = messages[-1].content.strip()
    return messages


def gen_unit_tests(ai: AI, dbs: DBs, project_prompt: str = None) -> List[dict]:
    
    print("Running gen_unit_tests")
    """
    Generate unit tests based on the specification, that should work.
    """
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {project_prompt}"),
        ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
    ]

    messages = ai.next(messages, dbs.preprompts["unit_tests"], step_name=curr_fn())

    dbs.memory["unit_tests"] = messages[-1].content.strip()
    to_files(dbs.memory["unit_tests"], dbs.workspace)   

    # Original Implementation
        # """
        # Generate unit tests based on the specification, that should work.
        # """
        # messages = [
        #     ai.fsystem(setup_sys_prompt(dbs)),
        #     ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        #     ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
        # ]

        # messages = ai.next(messages, dbs.preprompts["unit_tests"], step_name=curr_fn())

        # dbs.memory["unit_tests"] = messages[-1].content.strip()
        # to_files(dbs.memory["unit_tests"], dbs.workspace)

    return messages


def gen_clarified_code(ai: AI, dbs: DBs, project_prompt: str = None) -> List[dict]:
    """Takes clarification and generates code"""
    messages = AI.deserialize_messages(dbs.logs[clarify.__name__])

    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
    ] + messages[1:]
    messages = ai.next(messages, dbs.preprompts["use_qa"], step_name=curr_fn())

    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def gen_code(ai: AI, dbs: DBs, project_prompt: str = None) -> List[dict]:
    print("Running gen_code")
    # get the messages from previous step
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {project_prompt}"),
        ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
        ai.fuser(f"Unit tests:\n\n{dbs.memory['unit_tests']}"),
    ]
    messages = ai.next(messages, dbs.preprompts["use_qa"], step_name=curr_fn())
    to_files(messages[-1].content.strip(), dbs.workspace)

    # Original Implementation
        # # get the messages from previous step
        # messages = [
        #     ai.fsystem(setup_sys_prompt(dbs)),
        #     ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        #     ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
        #     ai.fuser(f"Unit tests:\n\n{dbs.memory['unit_tests']}"),
        # ]
        # messages = ai.next(messages, dbs.preprompts["use_qa"], step_name=curr_fn())
        # to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def execute_entrypoint(ai: AI, dbs: DBs, project_prompt: str = None, handle_errors: bool = True, iteration: int = 0) -> List[dict]:

    print("Executing the code...")
    error_log = []

    # Open a subprocess and redirect stderr to capture errors
    with subprocess.Popen("bash run.sh", shell=True, cwd=dbs.workspace.path, stderr=subprocess.PIPE) as p:
        try:
            # Wait for the process to complete and returns stdout, stderr
            _, stderr = p.communicate() 
        except KeyboardInterrupt:
            print()
            print("Stopping execution.")
            print("Execution stopped.")
            p.kill()
            print()

        if stderr:
            error_log = stderr.decode().splitlines() 

    # Write the error log if there are any errors
    if handle_errors and error_log:
        print(f"Execution error(s) detected")
        error_log_path = os.path.join(dbs.workspace.path, "error_log.txt") 

        with open(error_log_path, 'w') as file:
            file.writelines(line + '\n' for line in error_log) 

        if iteration < MAX_ITERATIONS:
            print(f"Attempting to fix the errors...")
            fix_execution_errors(ai, dbs, project_prompt, iteration + 1)
        else:
            print(f"Failed to fix the errors after {MAX_ITERATIONS} attempts")

    return []


def gen_entrypoint(ai: AI, dbs: DBs, project_prompt: str = None) -> List[dict]:
    messages = ai.start(
        system=(
            "You will get information about a codebase that is currently on disk in "
            "the current folder.\n"
            "From this you will answer with code blocks that includes all the necessary "
            "unix terminal commands to "
            "a) install dependencies "
            "b) run all necessary parts of the codebase (in parallel if necessary).\n"
            "Do not install globally. Do not use sudo.\n"
            "Do not explain the code, just give the commands.\n"
            "Do not use placeholders, use example values (like . for a folder argument) "
            "if necessary.\n"
        ),
        user="Information about the codebase:\n\n" + dbs.workspace["all_output.txt"],
        step_name=curr_fn(),
    )
    print()

    regex = r"```\S*\n(.+?)```"
    matches = re.finditer(regex, messages[-1].content.strip(), re.DOTALL)
    dbs.workspace["run.sh"] = "\n".join(match.group(1) for match in matches)
    return messages


def use_feedback(ai: AI, dbs: DBs):
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        ai.fassistant(dbs.workspace["all_output.txt"]),
        ai.fsystem(dbs.preprompts["use_feedback"]),
    ]
    messages = ai.next(messages, dbs.input["feedback"], step_name=curr_fn())
    to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


def fix_code(ai: AI, dbs: DBs, project_prompt: str = None):
    print("Running fix_code")
    messages = AI.deserialize_messages(dbs.logs[gen_code.__name__])
    execution_errors = check_execution_errors(dbs)
    code_output = messages[-1].content.strip()
    messages = [
        ai.fsystem(setup_sys_prompt(dbs)),
        ai.fuser(f"Instructions: {project_prompt}"),
        ai.fuser(code_output),
        ai.fuser(f"Execution error(s) detected: {execution_errors}"),
        ai.fsystem(dbs.preprompts["fix_code"]),
    ]
    messages = ai.next(
        messages, "Please fix any errors in the code above.", step_name=curr_fn()
    )
    to_files(messages[-1].content.strip(), dbs.workspace)

    # Original Implementation
        # messages = AI.deserialize_messages(dbs.logs[gen_code.__name__])
        # code_output = messages[-1].content.strip()
        # messages = [
        #     ai.fsystem(setup_sys_prompt(dbs)),
        #     ai.fuser(f"Instructions: {dbs.input['prompt']}"),
        #     ai.fuser(code_output),
        #     ai.fsystem(dbs.preprompts["fix_code"]),
        # ]
        # messages = ai.next(
        #     messages, "Please fix any errors in the code above.", step_name=curr_fn()
        # )
        # to_files(messages[-1].content.strip(), dbs.workspace)
    return messages


# New Functions to support the new execution flow

def check_execution_errors(dbs: DBs) -> List[str]:
    errors = []
    error_log_path = os.path.join(dbs.workspace.path, "error_log.txt") 
    
    if os.path.exists(error_log_path):
        with open(error_log_path, 'r') as file:
            lines = file.readlines()
            if lines:
                errors.append(lines[-1].strip())

    return errors

def fix_execution_errors(ai: AI, dbs: DBs, project_prompt: str = None, iteration: int = 0) -> List[dict]:
    print(f"Execution attempt {iteration}")
    execution_errors = check_execution_errors(dbs) # Function to check if there are any execution errors

    if not execution_errors:
        print("Execution successful")
        return []

    print(f"Execution error(s) detected: {execution_errors}")
    print("Attempting to fix the error...")

    # Apply the necessary fixes. This might involve calling existing functions like fix_code or other repair logic
    if not fix_code(ai, dbs, project_prompt): # Function to apply the necessary fixes
        print(f"Failed to fix the errors after {iteration} attempts")
        return []

    # If fixing was successful, execute the code again
    execute_entrypoint(ai, dbs, project_prompt, iteration=iteration)
    return []

def apply_fixes(ai: AI, dbs: DBs, execution_errors: List[str]) -> bool:

    for error in execution_errors:
        print(f"Attempting to fix error: {error}")

    return True

def human_review(ai: AI, dbs: DBs, project_prompt: str = None) -> List[dict]:
    review = human_input()
    dbs.memory["review"] = review.to_json()  # type: ignore
    return []


class Config(str, Enum):
    DEFAULT = "default"
    BENCHMARK = "benchmark"
    SIMPLE = "simple"
    TDD = "tdd"
    TDD_PLUS = "tdd+"
    CLARIFY = "clarify"
    RESPEC = "respec"
    EXECUTE_ONLY = "execute_only"
    EVALUATE = "evaluate"
    USE_FEEDBACK = "use_feedback"


# Different configs of what steps to run
STEPS = {
    Config.DEFAULT: [
        clarify,
        gen_clarified_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.BENCHMARK: [simple_gen, gen_entrypoint],
    Config.SIMPLE: [simple_gen, gen_entrypoint, execute_entrypoint],
    Config.TDD: [
        gen_spec,
        gen_unit_tests,
        gen_code,
        gen_entrypoint,
        execute_entrypoint,
        # human_review,
    ],
    Config.TDD_PLUS: [
        gen_spec,
        gen_unit_tests,
        gen_code,
        fix_code,
        gen_entrypoint,
        execute_entrypoint,
        # human_review,
    ],
    Config.CLARIFY: [
        clarify,
        gen_clarified_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.RESPEC: [
        gen_spec,
        respec,
        gen_unit_tests,
        gen_code,
        fix_code,
        gen_entrypoint,
        execute_entrypoint,
        human_review,
    ],
    Config.USE_FEEDBACK: [use_feedback, gen_entrypoint, execute_entrypoint, human_review],
    Config.EXECUTE_ONLY: [execute_entrypoint],
    Config.EVALUATE: [execute_entrypoint, human_review],
}

# Future steps that can be added:
# run_tests_and_fix_files
# execute_entrypoint_and_fix_files_if_it_results_in_error
