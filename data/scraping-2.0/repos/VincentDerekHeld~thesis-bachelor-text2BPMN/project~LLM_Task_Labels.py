import logging
import os

import requests
from openai import OpenAI

from project.LLM_API import generate_response_GPT3_instruct_model, generate_response_GPT4_model
import re


def remove_as_from_syntax(syntax: str) -> str:
    """ Remove the word 'as' from the syntax within square brackets (which represent tasks),
    as it leads to errors in the rendering of the BPMN diagram.
    Args:
        syntax: Syntax of the BPMN diagram
    Returns:
        Syntax of the BPMN diagram with 'as' removed from tasks
    """
    # Define a pattern to match '[', followed by any characters, 'as', any characters, and then ']'
    # The pattern captures the text before and after 'as' in separate groups
    # Replace 'as' with 'like' within (and only within) square brackets
    syntax = re.sub(r"(\[.*?)(\bas\b)(.*?\])", r"\1like\3", syntax)
    return syntax


if __name__ == '__main__':
    # Example usage
    text = "This is a test [with as] and [without]."
    print(remove_as_from_syntax(text))


def improve_taks_labeles_LLM(syntax: str, text_description: str) -> str:
    debug_mode = True
    prompt = (f"""
    Please solve the tasks carefully step by step.
    Tasks:
    1. Please do not change the title of the process diagram (first line).
    2. Please improve the texts of the tasks in the following process diagram carefully based on the given text description. Tasks are denoted with []. E.g. [task1] as activity_1. 
    3. If two tasks are described with "the first activity" and are based on the same object they can be merged.
    4. Please resolve imprecise descriptions such as "doing their tasks" carefully, based on the process description.
    5. Ensure each task is described with a maximum of 6 words and verbs are used in the base form.
    6. Please return carefully only the improved process diagram syntax as a simple string. No additional text is needed.
   
    Diagram Syntax to be Improved:
    {syntax}

    ### Full Process Description: ###
    {text_description}

    Note: Return only the improved process diagram syntax as a simple string. No additional text is needed.
    """)

    print(f"prompt: {prompt}")
    result = generate_response_GPT3_instruct_model(prompt)
    result = remove_as_from_syntax(result)
    print("**** Full description: **** \n" + result)
    return result.strip()


def improve_syntax_tasks(syntax: str, text_description: str) -> str:
    ### This worked pretty good
    debug_mode = True
    prompt = (f"""
    
    Tasks:
    1. Improve the texts of the tasks in the following process diagram based on the given text description. Tasks are denoted with []. E.g. [task1] as activity_1.
    2. Return only the improved process diagram syntax as a simple string. No additional text is needed.
    
    Diagram Syntax to be Improved:
    {syntax}

    ### Full Process Description: ###
    {text_description}

    Note: Return only the improved process diagram syntax as a simple string. No additional text is needed.
    """)

    print(f"prompt: {prompt}")
    result = generate_response_GPT3_instruct_model(prompt)
    print("**** Full description: **** \n" + result)
    return result.strip()


def improve_syntax(syntax: str, text_description: str) -> str:
    debug_mode = True
    prompt = (f"""
    Tasks: 
    1. Improve the syntax of the following process diagram based on the given text description and ensure that all elements are accurately described. 
    2. Carefully check if the syntax is correct and according the provided rules.
    3. Return only the improved process diagram syntax as a simple string. No additional text is needed.

    Rules:
    1. The first line must contain the word 'title'. Do not change the title.
    2. Start events are denoted by (start) E.g. (start) as start1
    3. End events  are denoted by (end). E.g. (end) as end1
    4. Tasks are denoted  with []. E.g. [task1] as activity_1
    5. Exclusive gateways are indicated with <>, and conditions within the <>. E.g. <> as gateway_1
    6. Parallel gateways are denoted by <@parallel>. E.g. <@parallel> as gateway_2
    7. Each branching gateway should be followed by a corresponding closing gateway.
    8. Assign a unique ID to all elements, including events, tasks, and gateways. E.g., activity_1, activity_2, gateway_1, gateway_2, etc.
    9. Elements are grouped into lanes based on the actor executing them.
    10. Each lane name must be unique and appear only once, except for unnamed single lanes. Each lane must contain at least one element.
    11. Connect events in lanes using '->'.
    12. For conditional gateways, annotate conditions like gateway_4-"condition"->activity_6->gateway_4_end.

    ### Example Input: ###
    A customer brings in a defective computer, and the CRS checks the defect and provides a repair cost calculation. If the costs are acceptable, the process continues; otherwise, the computer is returned unrepaired. The repair involves two activities executed in any order: hardware check and repair, and software check and configuration. Each activity is followed by a system functionality test. If an error is detected, another repair activity is executed; otherwise, the repair is finished.

    ### Example Output: ###
     Example Output:
    lane: customer
        (start) as start
        [brings a defective computer] as activity_9
        [takes her computer] as activity_4
        <> as gateway_1_end
        [execute two activities] as activity_12
        [the first activity check the hardware] as activity_13
        [the first activity repair the hardware] as activity_14
        [the second activity checks the software] as activity_15
        [the second activity configure the software] as activity_16
        [test the proper system functionality] as activity_17
        <detect an error?> as gateway_5
        [execute another arbitrary repair activity] as activity_7
        [finish the repair] as activity_8
        <> as gateway_5_end
        (end) as end
    lane: crs
        [checks the defect] as activity_10
        [hand out a repair cost calculation] as activity_11
        <the costs are acceptable?> as gateway_1
        [the process continues] as activity_3
    
    start->activity_9->activity_10->activity_11->gateway_1
    gateway_1-"yes"->activity_3->gateway_1_end
    gateway_1-"no"->activity_4->gateway_1_end
    gateway_1_end->activity_12->activity_13->activity_14->activity_15->activity_16->activity_17->gateway_5
    gateway_5-"yes"->activity_7->gateway_5_end
    gateway_5-"no"->activity_8->gateway_5_end
    gateway_5_end->end

    Diagram Syntax to be Improved:
    {syntax}

    ### Full Process Description: ###
    {text_description}

    Note: Return only the improved process diagram syntax as a simple string. No additional text is needed.
    """)

    print(f"prompt: {prompt}")
    result = generate_response_GPT3_instruct_model(prompt)
    print("**** Full description: **** \n" + result)
    return result


def improve_syntax_old(syntax: str, text_description: str) -> str:
    debug_mode = True
    prompt = (f"""
    Based on the following text description and the provided rules, please carefully improve the syntax of the following process diagram.
    Carefully ensure, that all elements are precisely described and that the syntax is correct.
    Here are some rules:
    1. The first line must contain the word 'title'
    2. the start event is always displayed with (start), and end events are always displayed with (end).
    3. tasks are indicated with a [],
    4. Exclusive gateways are indicated with <>, where the conditions can be specified within the <>
    5. The Parallel gateways are specified with <@parallel>.
    6. Every gateway that begins a branch should be followed by the same gateway that ends the branch.
    7. An unique ID will follow all these elements.
    8. Based on the actor who carried out the corresponding elements, the elements will be added to corresponding lanes.
    9. Each lane with the same name can only appear once. If there is only one lane, it can have no name.
    10. After all events are registered in the lanes, they will be then connected using "->".
    11. For conditional gateways, if there should be some conditional specifications, it can be annotated like gateway_4-"the part is available in house"->activity_6->gateway_4_end.
    

    ### Example Input: ###
    A customer brings in a defective computer and the CRS checks the defect and hands out a repair cost calculation back. If the customer decides that the costs are acceptable, the process continues, otherwise she takes her computer home unrepaired. The ongoing repair consists of two activities, which are executed, in an arbitrary order. The first activity is to check and repair the hardware, whereas the second activity checks and configures the software. After each of these activities, the proper system functionality is tested. If an error is detected another arbitrary repair activity is executed, otherwise the repair is finished.
    
    Example Output:
    lane: customer
        (start) as start
        [brings a defective computer] as activity_9
        [takes her computer] as activity_4
        <> as gateway_1_end
        [execute two activities] as activity_12
        [the first activity check the hardware] as activity_13
        [the first activity repair the hardware] as activity_14
        [the second activity checks the software] as activity_15
        [the second activity configure the software] as activity_16
        [test the proper system functionality] as activity_17
        <detect an error?> as gateway_5
        [execute another arbitrary repair activity] as activity_7
        [finish the repair] as activity_8
        <> as gateway_5_end
        (end) as end
    lane: crs
        [checks the defect] as activity_10
        [hand out a repair cost calculation] as activity_11
        <the costs are acceptable?> as gateway_1
        [the process continues] as activity_3
    
    start->activity_9->activity_10->activity_11->gateway_1
    gateway_1-"yes"->activity_3->gateway_1_end
    gateway_1-"no"->activity_4->gateway_1_end
    gateway_1_end->activity_12->activity_13->activity_14->activity_15->activity_16->activity_17->gateway_5
    gateway_5-"yes"->activity_7->gateway_5_end
    gateway_5-"no"->activity_8->gateway_5_end
    gateway_5_end->end
    
    Make sure to return only the syntax of the process diagram as a simple string, without any additional text .
    ### Activity: ###
    {text_description}

    ### Full Process Description: ###
    {syntax}
    """)
    print(f"prompt: {prompt}")
    result = ""
    result = generate_response_GPT4_model(prompt)
    print("**** Full description: **** \n" + result)
    return result
