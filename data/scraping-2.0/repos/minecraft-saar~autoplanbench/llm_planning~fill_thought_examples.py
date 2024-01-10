import os
import json
import openai
import re
from model_classes.planning_game_models import create_llm_model
from set_env import set_env_vars
from pathlib import Path
from argparse import ArgumentParser
from utils.paths import THOUGHT_GEN_EXAMPLE_FILE, THOUGHT_GEN_EXAMPLE_DOMAIN
from utils.helpers import get_llm_type
set_env_vars()
openai.api_key = os.environ['OPENAI_API_KEY']


def generate_reasoning_thoughts(template_file: str, nl_domain_file: str, example_nl_domain_file: str, react_example: str, llm: str, llm_type: str):
    """

    :param template_file:
    :param nl_domain_file:
    :param example_nl_domain_file:
    :param react_example:
    :param llm:
    :param llm_type:
    :return:
    """

    with open(template_file, 'r') as f:
        template = json.load(f)
        template = template['pos_examples'][0][0]

    with open(react_example, 'r') as f:
        react_example = json.load(f)
        react_example = react_example['pos_examples'][0][0]

    domain_intro = create_domain_intro(nl_domain_file)
    example_domain_intro = create_domain_intro(example_nl_domain_file)

    # Create the 1-shot example
    example_lines = react_example.split('\n')
    new_example_lines = []
    reasoning_steps = []
    for ex_l in example_lines:
        if ex_l.startswith('\tThink'):
            reasoning_steps.append(ex_l.strip())
            new_l = '\tThink: [TODO: ADD REASONING THOUGHT]'
            new_example_lines.append(new_l)
        else:
            new_example_lines.append(ex_l)
    input_example = '\n'.join(new_example_lines)
    # add domain description
    input_example = example_domain_intro + '\n' + input_example

    output_example = f'The following are good reasoning thoughts for the TODOs:\n'
    for step_id, step in enumerate(reasoning_steps):
        output_example += f'{step_id}. {step}\n'

    # Create model input
    model_input = domain_intro + '\n' + template

    system_prompt = """You are a brilliant and helpful assistant to provide reasoning thoughts for an interaction where one agent instructs another agent to execute a task.

You will be shown the actions that can be carried out, their preconditions and their effects. 

Additionally, you will see one interaction between the instruction giver and the instruction follower.

Your task is to come up with an appropriate and good reasoning thought with which [TODO: ADD REASONING THOUGHT] should be replaced. """

    model_param = {'model_path': llm,
                   'max_tokens': 300,
                   'temp': 0.0,
                   'max_history': 0}
    llm_model = create_llm_model(model_type=llm_type, model_param=model_param)
    llm_model.init_model(init_prompt=system_prompt)
    llm_model.add_examples(
        examples=[
            {"role": "user", "content": input_example},
            {"role": "assistant", "content": output_example}
        ])

    response = llm_model.generate(user_message=model_input)
    print(f'{llm_model.get_initial_prompt()}')
    print(input_example)
    print(output_example)
    print(f'{model_input}')
    print(f'{response}')

    response_parts = response.split('\n')
    relevant_responses = []
    for rp in response_parts:
        if 'Think:' in rp:
            relevant_responses.append(rp)

    reg = r'\[TODO: ADD REASONING THOUGHT\]'
    open_spaces = re.findall(reg, template)
    assert len(open_spaces) == len(relevant_responses)

    final_few_shot_example = template
    for resp in relevant_responses:
        # replace the occurrences of [TO DO ...] with the responses from left to right
        cleaned_resp = resp.split('Think:')[-1]
        cleaned_resp = cleaned_resp.strip()
        final_few_shot_example = re.sub(pattern=reg, repl=cleaned_resp, string=final_few_shot_example, count=1)

    return final_few_shot_example


def create_react_few_shot_file(template_file: str, nl_domain_file: str, example_nl_domain_file: str, react_example: str, output_file: str, llm: str, llm_type: str):
    """

    :param template_file:
    :param nl_domain_file:
    :param example_nl_domain_file:
    :param react_example:
    :param output_file:
    :param llm:
    :param llm_type:
    :return:
    """
    output_dir = os.path.split(output_file)[0]
    Path(output_dir).mkdir(exist_ok=True)

    react_few_shot = generate_reasoning_thoughts(template_file=template_file, nl_domain_file=nl_domain_file,
                                example_nl_domain_file=example_nl_domain_file, react_example=react_example, llm=llm, llm_type=llm_type)

    with open(output_file, 'w') as f:
        example_dict = {
            "prefixes": {"input": "", "output": ""},
            "pos_examples": [[react_few_shot, '']]
        }
        json.dump(example_dict, f, indent=4)


def create_cot_few_shot_file(template_file: str, nl_domain_file: str, example_nl_domain_file: str, react_example: str, output_file: str, llm: str, llm_type: str):
    """

    :param template_file:
    :param nl_domain_file:
    :param example_nl_domain_file:
    :param react_example:
    :param output_file:
    :param llm:
    :param llm_type:
    :return:
    """
    output_dir = os.path.split(output_file)[0]
    Path(output_dir).mkdir(exist_ok=True)

    filled_react_example = generate_reasoning_thoughts(template_file=template_file, nl_domain_file=nl_domain_file,
                                                       example_nl_domain_file=example_nl_domain_file,
                                                       react_example=react_example, llm=llm, llm_type=llm_type)

    create_cot_from_react(output_file=output_file, final_react_example=filled_react_example)




def create_cot_from_react(output_file: str, final_react_example: str):
    """

    :param output_file:
    :param final_react_example:
    :return:
    """
    output_dir = os.path.split(output_file)[0]
    Path(output_dir).mkdir(exist_ok=True)

    example_lines = final_react_example.split('\n')
    filled_cot_example_input = []
    filled_cot_example_output = []

    plan_begin = False
    for line in example_lines:
        # everything before the first 'You' / model turn is part of the problem description
        if line.startswith('You:'):
            plan_begin = True
            continue
        elif not plan_begin:
            line = line.replace('I: ', '')
            filled_cot_example_input.append(line)
        # lines containing only 'You'
        # remove observations from environment
        elif line.startswith('I: '):
            continue
        else:
            line = line.strip()
            filled_cot_example_output.append(line)

    filled_cot_example_input.append("Let's think step by step")
    filled_cot_example_output.append('[PLAN END]')

    with open(output_file, 'w') as f:
        example_dict = {
            "prefixes": {"input": "[STATEMENT]", "output": "[PLAN]"},
            "pos_examples": [['\n'.join(filled_cot_example_input), '\n'.join(filled_cot_example_output)]]
        }
        json.dump(example_dict, f, indent=4)




def create_domain_intro(nl_domain_file: str):

    with open(nl_domain_file, 'r') as f:
        domain_nl = json.load(f)

    if 'domain_intro' in domain_nl:
        domain_intro = domain_nl['domain_intro']
        domain_intro = domain_intro.replace("I have to plan logistics to transport packages within cities via trucks and between cities via airplanes. Locations within a city are directly connected (trucks can move between any two such locations), and so are the cities. In each city there is exactly one truck and each city has one location that serves as an airport.\n", "")
        domain_intro = domain_intro.replace("I am playing with a set of blocks where I need to arrange the blocks into stacks. ", "")

    else:
        action_descrp = 'I can carry out the following actions:\n'
        prec_descrp = 'I have the following restrictions on my actions:\n'
        effect_descr = 'The actions have the following effects on the state:\n'
        for action, action_dict in domain_nl['actions'].items():
            action_d = action_dict['description']
            action_descrp += f'{action_d}\n'
            for pr in action_dict['preconditions']:
                prec_descrp += f'{pr}\n'
            for ef in action_dict['effects']:
                effect_descr += f'{ef}\n'

        domain_intro = f'{action_descrp}\n{prec_descrp}\n{effect_descr}'

    return domain_intro



def create_react_and_cot(template_file: str, nl_domain_file: str, example_nl_domain_file: str, react_example: str, react_output_file: str, llm: str, llm_type: str):
    """

    :param template_file:
    :param nl_domain_file:
    :param example_nl_domain_file:
    :param react_example:
    :param react_output_file:
    :param llm:
    :param llm_type:
    :return:
    """
    output_dir = os.path.split(react_output_file)[0]
    Path(output_dir).mkdir(exist_ok=True)

    create_react_few_shot_file(template_file=template_file, nl_domain_file=nl_domain_file,
                               example_nl_domain_file=example_nl_domain_file,
                               react_example=react_example, llm=llm,
                               output_file=react_output_file, llm_type=llm_type)


    with open(react_output_file, 'r') as f:
        react_few_shot = json.load(f)
    filled_react_example = react_few_shot['pos_examples'][0][0]

    cot_output_file = str(react_output_file).replace('react', 'cot')
    assert cot_output_file != react_output_file

    create_cot_from_react(output_file=cot_output_file, final_react_example=filled_react_example)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--template', required=True, help="path to the template file")
    parser.add_argument('--nl-domain', required=True, help="path to the file with the nl domain description")
    parser.add_argument('--ex-domain', required=False, default=THOUGHT_GEN_EXAMPLE_DOMAIN, help="path to the file with the nl description of the example domain; defaults to utils.paths.THOUGHT_GEN_EXAMPLE_DOMAIN")
    parser.add_argument('--ex-react', required=False, default=THOUGHT_GEN_EXAMPLE_FILE, help="path to the file with the react interaction example; defaults to utils.paths.THOUGHT_GEN_EXAMPLE_FILE")
    parser.add_argument('--out', required=True, help="path to the output file for the react example; the path to the output file for the "
                                                     "cot example is derived by replacing 'react' by 'cot'")

    parser.add_argument('--llm', required=True, help="name of the llm to use (currently only chat gpt models possible)")
    parser.add_argument('--llm_type', required=False, default=None, help='type of the llm to use')

    args = parser.parse_args()

    template = args.template
    nl_domain = args.nl_domain
    example_domain = args.ex_domain
    react_interaction_example = args.ex_react
    out_file = args.out
    llm = args.llm
    model_type = args.llm_type if args.llm_type is not None else get_llm_type(args.llm)


    create_react_and_cot(template_file=template, nl_domain_file=nl_domain,
                         example_nl_domain_file= example_domain, react_example=react_interaction_example,
                         react_output_file=out_file, llm=llm, llm_type=model_type)


