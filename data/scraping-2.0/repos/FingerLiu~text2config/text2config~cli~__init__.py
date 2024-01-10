import argparse
from argparse import RawTextHelpFormatter
from langchain.prompts import PromptTemplate
from text2config.pkg.openai import prompts
from text2config.pkg.openai import quest

parser = argparse.ArgumentParser(description="""Convert natural language text to configuration files(yaml/ini/conf/json) 
or command of various projects(docker/kubernetes/vim/nginx/postgres/terraform).

Example: t2c -e k8s "get all pod in namespace kube-system and sort by create time"
""",
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument("-m", "--mode", type=str, choices=["cmd", "config"],
                    default="cmd", help="generate command or config")
parser.add_argument("--command", "-e", type=str,
                    default="any", help="command name: docker, k8s, kubernetes, kubectl, nginx, any, ...")
parser.add_argument("goal", type=str, help="goal of the command or config that you want to generate")
parser.add_argument("-c", "--config")
parser.add_argument("-d", "--debug", action="store_true", default=False, help="show debug log")


def parse_args():
    args = parser.parse_args()
    command = args.command
    goal = args.goal
    mode = args.mode
    config = args.config
    if args.debug:
        print(f"command goal mode config: {command} {goal} {mode} {config}")
    if command not in ('docker', 'k8s', 'kubernetes', 'kubectl', 'nginx', 'any'):
        cmd_type = 'other'
    elif command in ('k8s', 'kubernetes', 'kubectl'):
        cmd_type = 'k8s'
    else:
        cmd_type = command
    prompt_str = f"{cmd_type}_{mode}"
    if hasattr(prompts, prompt_str):
        promptTemplate: PromptTemplate = getattr(prompts, prompt_str)
        if cmd_type == 'other':
            prompt = promptTemplate.format(goal=goal, cmd=command)
        else:
            prompt = promptTemplate.format(goal=goal)
        if args.debug:
            print(prompt)
        print(f'{quest(prompt)}\n')



