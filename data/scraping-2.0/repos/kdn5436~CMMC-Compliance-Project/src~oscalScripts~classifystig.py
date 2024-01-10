from openai import OpenAI
from parse_cmmc_catalog import get_control_llm_system_message
import os, dotenv
import yaml

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG_ID"))
dotenv.load_dotenv()


# L1_CONTROLS_SYSTEM_MSG = get_control_llm_system_message('../../ref/json/cmmc_v2_L1.json')
L1_CONTROLS_SYSTEM_MSG = open('l2_sys_msg_ctrls.txt', 'r').read()
# CLIENT = openai.Client(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG_ID"))
# def win10_task_relevant_info(task: dict, is_subtask: bool = False):
#     name = task["name"].split(" | ")[3]
#     if is_subtask:
#         name = " ".join(task["name"].split(" | ")[3:5])
#     task_type = task["name"].split(" | ")[2]
#     implementation_key = [key for key in task.keys() if key.startswith("ansible") and not "debug" in key]
#     implementation = yaml.dump({key:task[key] for key in implementation_key})
#     return name, task_type, implementation


def win10_task_relevant_info(task: dict):
    if "\n" in task["name"]:
        name = "; ".join(map(lambda x: x.split(" | ")[3], task["name"].replace('"', "").replace("'", "")[:-1].split("\n")))
        # TODO below uses the same type as the first for each. hacky but should be fine
        id = task["name"].split("\n")[0].split(" | ")[1]
        task_type = task["name"].split("\n")[0].split(" | ")[2]
    else:
        name = task["name"].split(" | ")[3]
        id = task["name"].split(" | ")[1]
        task_type = task["name"].split(" | ")[2]
    if "block" in task.keys():
        implementation = yaml.dump([{key: val for key, val in subtask.items() if key != vars} for subtask in task["block"]])
    else:
        implementation_key = [key for key in task.keys() if key != "when" and key != "tags" and key != "name"]
        implementation = yaml.dump({key:task[key] for key in implementation_key})

    # print(f"Task ID: {id}")
    # print(f"Task Name: {name}")
    # print(f"Task Type: {task_type}")
    # print(f"Task Implementation: {implementation}")
    
    return id, name, task_type, implementation

def classify_control(name: str, implementation: str):
#     sys_msg_content = f"""You are a component in a program that maps an ansible security implementation to a relevant security control in compliance with the CMMC Model V2 Security controls.
# You will be given an ansible task that implements a security control, and you will be respond with the ID of the security control that best represents the ansible task.
# Below is a list of CMMC Model V2 Level 1 Security Controls.  For each control, there is it's ID and Title, followed by a description of what the control requires.
# Controls:
# ```
# {L1_CONTROLS_SYSTEM_MSG}
# ```
# If the provided task is not relevant to any of the controls, respond with the word 'NONE'."""
    sys_msg_content = open('sys_msg.txt', 'r').read()
    sys_msg = [{"role": "system", "content": sys_msg_content}]
    task_content = f"""Task Description:
    {name}
    Task Implementation:
    {implementation}
    """
    print(sys_msg[0]["content"].replace("\n\n", "\n"))
    # print("------")
    # print(task_content)
    return
    task_msg = [{"role": "user", "content": task_content}]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=0.0,
        messages=sys_msg + task_msg,
    )

    gpt_output = response.choices[0].message.content

    return gpt_output, response.usage.total_tokens

def classify_controls_in_task_win10(task_filepath: str):
    classifications = []
    used_tokens = 0
    with open(task_filepath, "r") as f:
        tasks = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        manual_tasks = []
        automated_tasks = []

        for task in tasks:
            #TODO PATCH tasks have changing actions, and thus can be classified based on the action (ansible.windows.win_regedit, ansible.windows.win_shell, etc...)
            # AUDIT tasks can either have a verification action, or a when conditional based on a previous action in the block
            # tasks are nested if they have a block key, which contains a list of more tasks
            if "block" in task.keys():
                #  for subtask in task["block"]:
                # name, task_type, implementation = win10_task_relevant_info(task, is_subtask=True)
                id, name, task_type, implementation = win10_task_relevant_info(task)
                if task_type == "AUDIT":
                    manual_tasks.append(task)
                else:
                    automated_tasks.append(task)

                

            # output, tokens = classify_control(name, implementation)
            # classifications.append({id: output})
            # used_tokens += tokens

    return classifications, used_tokens

def main():
    # classifications, tokens_used = classify_controls_in_task_win10("../stiglevel2/roles/Win10STIG/tasks/cat3.yml")

    with open("testyaml.yml", 'r') as f:
        tasks = yaml.load(f.read(), Loader=yaml.FullLoader)
        for task in tasks:
            id, name, typ, impl = win10_task_relevant_info(task)
            # classify_control(name, impl)
            print(name)
            print("---")
            print(impl)
            print("------")
            # print(L1_CONTROLS_SYSTEM_MSG)
            # classify_control(name, impl)

if __name__ == '__main__':
    main()