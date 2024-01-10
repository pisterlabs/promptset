import openai




with open('openai_api_key.txt', 'r') as file:
    OPENAI_API_KEY = file.read().strip()
openai.api_key = OPENAI_API_KEY

color_prefix_by_role = {
    "system": "\033[0m",  # gray
    "user": "\033[0m",  # gray
    "assistant": "\033[92m",  # green
}


def print_messages(messages, color_prefix_by_role=None) -> None:
    """Prints messages sent to or from GPT, with optional custom roles and colors."""
    if color_prefix_by_role is None:
        color_prefix_by_role = {
            "system": "\033[0m",  # gray
            "user": "\033[0m",    # gray
            "assistant": "\033[92m",  # green
        }
    for message in messages:
        if "role" not in message or "content" not in message:
            print("Invalid message format: role and content required.")
            continue
        role = message["role"]
        content = message["content"]
        color_prefix = color_prefix_by_role.get(role, "\033[0m")  # Default to gray if role is unknown
        print(f"{color_prefix}\n[{role}]\n{content}")



def print_message_delta(delta, color_prefix_by_role=color_prefix_by_role) -> None:
    """Prints a chunk of messages streamed back from GPT."""
    if "role" in delta:
        role = delta["role"]
        color_prefix = color_prefix_by_role[role]
        print(f"{color_prefix}\n[{role}]\n", end="")
    elif "content" in delta:
        content = delta["content"]
        print(content, end="")
    else:
        pass




def unit_tests_from_function(
    function_to_test: str,  # Python function to test, as a string
    unit_test_package: str = "pytest",  # unit testing package; use the name as it appears in the import statement
    approx_min_cases_to_cover: int = 4,  # minimum number of test case categories to cover (approximate)
    print_text: bool = True,  # optionally prints text; helpful for understanding the function & debugging
    explain_model: str = "gpt-4-0613",  # model used to generate text plans in step 1
    plan_model: str = "gpt-4-0613",  # model used to generate text plans in steps 2 and 2b
    execute_model: str = "gpt-4-0613",  # model used to generate code in step 3
    temperature: float = 0.4,  # temperature = 0 can sometimes get stuck in repetitive loops, so we use 0.4
    reruns_if_fail: int = 1,  # if the output code cannot be parsed, this will re-run the function up to N times
) -> str:
    """Returns a Task list to create a tool, using a 3-step GPT prompt."""

    # Step 1: Generate an explanation of the function

    # create a markdown-formatted message that asks GPT to explain the function, formatted as a bullet list
    explain_system_message = {
        "role": "system",
        "content": (
            "You are a world-class task planner following the methodology from the book 'The Task List Manifesto'. "
            "You create coherent explanations and steps of a plan to accomplish a goal. You explain the goal "
            "and the steps to accomplish the goal in fine detail. You organize your explanation in "
            "markdown-formatted, bulleted lists.")
    }

    explain_user_message = {
        "role": "user",
        "content": f"""Please create a detailed task list for the goal described below. Ensure each step is clearly defined and logically ordered. Organize your task list as a markdown-formatted, bulleted list.

    {function_to_test}
    """
    }

    explain_messages = [explain_system_message, explain_user_message]
    if print_text:
        print_messages(explain_messages)

    explanation_response = openai.ChatCompletion.create(
        model=explain_model,
        messages=explain_messages,
        temperature=temperature,
        stream=True,
    )
    explanation = ""
    for chunk in explanation_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            explanation += delta["content"]
    explain_assistant_message = {"role": "assistant", "content": explanation}
    # Step 2: Generate a plan for creating a tool for a goal

    # Asks GPT to plan out tasks that should be completed to create the tool, formatted as a bullet list
    plan_user_message = {
        "role": "user",
        "content": f"""To successfully create a tool that serves our goal, a good task list should:
    - Clearly define the objective of the tool.
    - Specify the functionalities and features it should have.
    - Identify potential challenges and how to overcome them.
    - Highlight any dependencies or prerequisites.
    - Be organized in a logical sequence for execution.

    Given the goal described above, please provide a detailed task list to guide the creation of the users desired python script. Organize your list as a markdown-formatted, bulleted list, and under each main task, include sub-tasks or further explanations as sub-bullets. No tests this is for an MVP of the python script for the user to run on their machine.""",
    }
    plan_messages = [
        explain_system_message,
        explain_user_message,
        explain_assistant_message,
        plan_user_message,
    ]
    if print_text:
        print_messages([plan_user_message])
    plan_response = openai.ChatCompletion.create(
        model=plan_model,
        messages=plan_messages,
        temperature=temperature,
        stream=True,
    )
    plan = ""
    for chunk in plan_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            plan += delta["content"]
    plan_assistant_message = {"role": "assistant", "content": plan}

    # Step 2b: If the plan is short, ask GPT to elaborate further
    # this counts top-level bullets (e.g., categories), but not sub-bullets (e.g., test cases)
    num_bullets = max(plan.count("\n-"), plan.count("\n*"))
    elaboration_needed = num_bullets < approx_min_cases_to_cover
    if elaboration_needed:
        elaboration_user_message = {
            "role": "user",
            "content": f"""Based on the task list, create a bulleted list of all the necessary modules and methods required, create a skeleton that will later be expanded, no tests, no refining, just write the code for an MVP).""",
        }
        elaboration_messages = [
            explain_system_message,
            explain_user_message,
            explain_assistant_message,
            plan_user_message,
            plan_assistant_message,
            elaboration_user_message,
        ]
        if print_text:
            print_messages([elaboration_user_message])
        elaboration_response = openai.ChatCompletion.create(
            model=plan_model,
            messages=elaboration_messages,
            temperature=temperature,
            stream=True,
        )
        elaboration = ""
        for chunk in elaboration_response:
            delta = chunk["choices"][0]["delta"]
            if print_text:
                print_message_delta(delta)
            if "content" in delta:
                elaboration += delta["content"]
        elaboration_assistant_message = {"role": "assistant", "content": elaboration}

    # Step 3: Refinement and Tool Creation

    # Create a markdown-formatted message to refine the task list and possibly provide a detailed tool or code.
    execute_system_message = {
        "role": "system",
        "content": (
            "You are a world-class method maker and python developer. Given the provided information, "
            "refine the python script and, if necessary, provide additional skeletons for classes, modules and methods to aid in accomplishing the goal of MVP for user to run on their system. "
            "Ensure every step or tool is precisely designed, aligning with the stated objectives."
        )
    }

    execute_user_message = {
        "role": "user",
        "content": f"""Based on the task list and plan provided, refine it further or create another version of the python tool skeleton.

    {plan}  # This incorporates the task list generated from the previous steps.
    """,
    }

    execute_messages = [
        execute_system_message,
        explain_user_message,
        explain_assistant_message,
        plan_user_message,
        plan_assistant_message,
    ]
    print(execute_messages)
    if elaboration_needed:
        execute_messages += [elaboration_user_message, elaboration_assistant_message]

    execute_messages += [execute_user_message]
    if print_text:
        print_messages([execute_system_message, execute_user_message])

    execute_response = openai.ChatCompletion.create(
        model=execute_model,  # Sticking with the "execute" theme.
        messages=execute_messages,
        temperature=temperature,
        stream=True,
    )

    execution = ""
    for chunk in execute_response:
        delta = chunk["choices"][0]["delta"]
        if print_text:
            print_message_delta(delta)
        if "content" in delta:
            execution += delta["content"]

    # For our purpose, checking for syntax errors in the generated execution is not needed.
    # But if you want to process or validate the refinement, you'd do it here.

    # Return the refined plan or tool as a string.
    return execution
