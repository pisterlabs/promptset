import os
import time
import random
import argparse


import openai

from deontic_domains.axiom_templates import *
from deontic_domains.prompts import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-4",
                        help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=400)
    parser.add_argument('--n_hops', type=int, default=2)
    parser.add_argument('--n_problems', type=int, default=1)
    parser.add_argument('--domain', type=str, default="triage")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--use_axioms', action='store_true')
    parser.add_argument('--load_problem', action='store_true')
    args = parser.parse_args()
    return args

def get_chat_response(prompt_messages, args):
    success = False
    while not success:
        try:
            response = openai.ChatCompletion.create(model=args.model, messages=prompt_messages,
                                                    temperature=args.temperature,
                                                    max_tokens=args.max_tokens)
            success = True
        except openai.error.RateLimitError:
            print("Error in chat response, retrying...")
            time.sleep(60)
    prediction = response['choices'][0]['message']['content']
    return prediction

def get_axioms(text):
    lines = text.split("\n")
    deontic_axioms = []
    theory_axioms = []
    for line in lines:
        if 'exists' in line or 'forall' in line:
            continue
        if 'daxiom' in line:
            deontic_axioms.append(line)
        elif 'taxiom' in line:
            deontic_axioms.append(line)
            
    deontic_axioms = "\n".join(deontic_axioms)
    theory_axioms = "\n".join(theory_axioms)
    return deontic_axioms, theory_axioms

def parse_problem(problem):
    # hacky way to parse the problem
    lines = problem.split("\n")
    context = []
    result = []
    res_id = 10000 
    deontic_axioms = []
    theory_axioms = []
    for l, line in enumerate(lines):
        if 'let daxiom' in line:
            deontic_axioms.append(line)
        elif 'let taxiom' in line:
            theory_axioms.append(line)
        elif 'let' in line:
            context.append(line)
        else:
            if 'Result' in line:
                res_id = l
            if l > res_id:
                result.append(line)
    context = "\n".join(context)
    deontic_axioms = "\n".join(deontic_axioms)
    theory_axioms = "\n".join(theory_axioms)            
    result = "\n".join(result)
    return context, deontic_axioms, theory_axioms, result

def copy_problem(problem, domain):
    copied_problem = domain.start_derivation(problem=problem.description, goal=None)
    copied_problem.universe = problem.universe.clone()
    return copied_problem

def dfs_search(dom_problem, domain, max_depth):
    stack = [(dom_problem, 0, [])]
    visited = set()


    while stack:
        cur_problem, depth, path = stack.pop()
        if args.verbose:
            print(f' Depth: {depth} len stack: {len(stack)}')
        # if (depth, cur_problem.universe) in visited:
        #     continue
        # visited.add((depth, cur_problem.universe))

        # Check if the last action is an axiom related action
        if depth == max_depth:
            if args.verbose:
                print('Found axiom')
            return path


        actions = domain.derivation_actions(cur_problem.universe)
        random.shuffle(actions)
        if depth < max_depth - 1:
            actions = [action for action in actions if 'taxiom' in action]
        elif depth == max_depth - 1:
            actions = [action for action in actions if 'daxiom' in action]

        for action in actions:
            temp_problem = copy_problem(cur_problem, domain)

            if depth == 0:
                outcomes = temp_problem.universe.apply(action)
            else:
                outcomes = temp_problem.universe.apply_with(action, f'r{depth}')

            if len(outcomes) == 0:
                continue

            random.shuffle(outcomes)
            for outcome in outcomes:
                temp_problem_o = copy_problem(temp_problem, domain)
                if args.verbose:
                    print(f'outcome: {outcome}')
                try: 
                    domain.define(temp_problem_o.universe, f'r{depth + 1}', outcome)
                except:
                    print(f'Error defining, {outcome}')
                    continue
                stack.append((temp_problem_o, depth + 1, path + [outcome, action]))
    return None

def exhaustive_search(problem, n_hops, domain):
    dom_problem = domain.start_derivation(problem=problem, goal=None)
    result = dfs_search(dom_problem, domain, n_hops)
    # result = dfs_search(dom_problem, domain, 0, n_hops, [])
    return result




if __name__ == "__main__":
    args = get_args()
    theory_file = f"../environment/theories/{args.domain}.p" 
    if args.domain == 'triage':
        from deontic_domains.triage_domain import TriageDomain, contexts, deontic_axioms, theory_axioms, domain_text
    elif args.domain == 'calendar':
        from deontic_domains.calendar_domain import CalendarDomain, contexts, deontic_axioms, theory_axioms, domain_text 
    else:
        raise NotImplementedError
    with open(theory_file, 'r') as f:
        theory = f.read()
    n_hops = args.n_hops
    n_problems = args.n_problems

    for n in range(n_problems):
        # sample context using gpt-4
        if not args.load_problem:
            system_context_dom = system_context.format(theory=theory)
            example_context_dom = example_context.format(context=contexts)
            system_axioms_dom = system_axioms[args.domain].format(theory=theory)
            axiom_templates_dom = axiom_templates.format(deontic_axioms=deontic_templates, theory_axioms=theory_templates)
            example_axioms = deontic_axioms.format(deontic_axiom=deontic_axioms, theory_axioms=theory_axioms)



            if args.use_context:
                # manual intervention for getting hops
                context = """let h4 : healthcare_professional.
let h5 : healthcare_professional.
let p3 : patient.
let p4 : patient.
let eq3 : equipment.
let eq4 : equipment.

let t3 : (physical_therapy p3).
let t4 : (medication p4).

let res3 : (low_resources h4).
let stab2 : (stable_condition p3).

let cons2 : (unconscious p3).
let mob3 : (mobile p4).

let inf1 : (infected p3).
let preg1 : (pregnant p4).
"""
            else:
                context_prompt = get_context_prompt(system_context_dom, example_context_dom)
                context = get_chat_response(context_prompt, args)
            if args.verbose:
                print(f"Context: {context}")

            axiom_prompt = get_axiom_prompt(system_axioms_dom, axiom_templates_dom, example_axioms, example_context_dom, context)

            if args.use_axioms:
                # manual intervention for getting n hops
                gen_deontic_axioms = """let daxiom11 : [('h : healthcare_professional) -> ('p : patient) -> (critical_condition 'p) -> (obligatory (perform_examination 'h 'p))].
let daxiom12 : [('h : healthcare_professional) -> ('p : patient) -> (stable_condition 'p) -> (obligatory (assign_green 'h 'p))].
let daxiom13 : [('h : healthcare_professional) -> ('p : patient) -> (infected 'p) -> (impermissible (discharge_patient 'h 'p))].
let daxiom14 : [('h : healthcare_professional) -> ('p : patient) -> (pregnant 'p) -> (permissible (initiate_rehabilitation 'h 'p))].
let daxiom15 : [('h : healthcare_professional) -> ('p : patient) -> (low_resources 'h) -> (not (obligatory (administer_treatment 'h 'p)))].
let daxiom16 : [('h : healthcare_professional) -> ('p : patient) -> (mobile 'p) -> (permissible (create_treatment_plan 'h 'p))].
let daxiom17 : [('h : healthcare_professional) -> ('e : equipment) -> (not_assigned_care 'h p3) -> (not (obligatory (use_equipment 'h p3 'e)))].
let daxiom18 : [('h : healthcare_professional) -> ('p : patient) -> (conscious 'p) -> (not (obligatory (discharge_patient 'h 'p)))].
let daxiom19 : [('h : healthcare_professional) -> ('p : patient) -> (unconscious 'p) -> (obligatory (monitor_patient 'h 'p))].
let daxiom20 : [('h : healthcare_professional) -> ('p : patient) -> (assigned_care 'h 'p) -> (obligatory (provide_family_support 'h 'p))].
"""
                gen_theory_axioms = """let taxiom1 : [('p : patient) -> (infected 'p) -> (unstable_vitals 'p)].
let taxiom2 : [('p : patient) -> (pregnant 'p) -> (unstable_vitals 'p)].
let taxiom3 : [('p : patient) -> (stable_condition 'p) -> (conscious 'p)].
let taxiom4 : [('h : healthcare_professional) -> ('p : patient) -> (conscious 'p) -> (not_assigned_care 'h 'p)].
let taxiom5 : [('h : healthcare_professional) -> ('p : patient) -> (high_resources 'h) -> (assigned_care 'h 'p)].
let taxiom6 : [('h : healthcare_professional) -> ('p : patient) -> (declare_emergency 'h) -> (critical_condition 'p)].
let taxiom7 : [('h : healthcare_professional) -> ('p : patient) -> (assess_mental_health 'h 'p) -> (unstable_mental_state 'p)].
let taxiom8 : [('h : healthcare_professional) -> ('p : patient) -> (provide_family_support 'h 'p) -> (stable_mental_state 'p)].
let taxiom9 : [('p : patient) -> (assign_orange h4 'p) -> (critical_condition 'p)].
let taxiom10 : [('p : patient) -> (unstable_vitals 'p) -> (assign_orange h4 'p)].
"""
            else:
                axiom_response = get_chat_response(axiom_prompt, args)
                gen_deontic_axioms, gen_theory_axioms = get_axioms(axiom_response) 

            if args.verbose:
                print(f"Deontic Axioms: {gen_deontic_axioms}")
                print(f"Theory Axioms: {gen_theory_axioms}")

            # define the context
            problem = f"{context}\n{gen_deontic_axioms}{gen_theory_axioms}"

            # define the theory
            if args.domain == "calendar":
                domain = CalendarDomain('calendar.p')
            elif args.domain == "triage":
                domain = TriageDomain('triage.p')
            

            # sample an outcome
            sampled_outcomes = []

            
            # Execute the exhaustive search
            result = exhaustive_search(problem, n_hops, domain)
            print("Exhaustive search result:", result)

            # store the problem, theory, context, outcome, and solution
            # check files to find the next problem id
            if result is not None:
                file_num = len([f for f in os.listdir("deontic_domains/") if f.endswith(".p") and f"{args.domain}_problem_{args.n_hops}" in f])
                
                # save the problem as problem_00.p
                with open(f"deontic_domains/{args.domain}_problem_{args.n_hops}_{file_num:02d}.p", 'w') as f:
                    f.write(problem)
                    f.write("\nResult:\n")
                    result_str = '\n'.join([str(r) for r in result[::2]])
                    f.write(result_str)
        else:
            peano_files = sorted([f for f in os.listdir("deontic_domains/") if f.endswith(".p") and f"{args.domain}_problem_{args.n_hops}" in f])
            text_files = sorted([f for f in os.listdir("deontic_domains/") if f.endswith(".txt") and f"{args.domain}_problem_{args.n_hops}"])
            # check which peano files have a corresponding text file
            problem_file = None
            example_file = []
            for f in peano_files:
                text_file = f.split(".")[0] + ".txt"
                if text_file not in text_files:
                    problem_file = f.split(".")[0]
                else:
                    example_file.append(f.split(".")[0])
            assert problem_file is not None, "No problem file found"        
            if len(example_file) == 0:
                example_file = [f'{args.domain}_problem_4_00']
            example_file = [f'{args.domain}_problem_4_00']


            # sample current problem
            with open(f"deontic_domains/{problem_file}.p", 'r') as f:
                problem = f.read()
                context, gen_deontic_axioms, gen_theory_axioms, result = parse_problem(problem)
            example = random.choice(example_file)
            with open(f"deontic_domains/{example}.p", 'r') as f:
                problem = f.read()
                example_context, example_deontic_axioms, example_theory_axioms, example_result = parse_problem(problem)
            # read example text
            with open(f"deontic_domains/{example}.txt", 'r') as f:
                example_text = f.read()
            
            if args.verbose:
                print("Context:", context)
                print("Deontic Axioms:", gen_deontic_axioms)
                print("Theory Axioms:", gen_theory_axioms)
                print("Result:", result)
            
            prompt = get_text_prompt(domain_text, theory, example_context, example_deontic_axioms+example_theory_axioms, example_result, example_text,
                                    context, gen_deontic_axioms+gen_theory_axioms, result)
            text = get_chat_response(prompt, args)

            with open(f"deontic_domains/{problem_file}.txt", 'w') as f:
                f.write(text)
            
