import os
import yaml
import argparse
import random
import openai

from Language2PDDL.scripts.inference import fewshot_evaluation, \
    constr_eval_goal_metric, constr_validate_plan_metric, constr_strict_eval_goal_metric, \
    constr_eval_list_goal_metric, constr_eval_lang_goal_metric, constr_eval_check_metric, \
    select_example_by_order, select_example_by_indices, select_example_by_shortest

def fewshot_test(args, config, mode, example=None):
    dataset_template = config["dataset_template"]
    n_obj_list = config["n_obj_list"]
    shuffled = config["shuffled"]
    uniform_dataset_configs = [ (n, s) for n in n_obj_list for s in shuffled ]
    selector = lambda path: select_example_by_order(path, 1)
    # selector = lambda path: select_example_by_indices(path, [1,])
    model = args.model
    for n, s in uniform_dataset_configs:
        example_split_name = dataset_template.format(obj=n, mode="train")
        if mode == "check":
            test_split_name = dataset_template.format(obj=n, mode="check")  
        else:
            
            test_split_name = dataset_template.format(obj=n, mode="test")        
        if mode == "goal":
            example_path = args.goal_ft_template.format(example_split_name)
            if example:
                example_path = example
            fewshot_evaluation(model=model, 
                                example_path=example_path,
                                dataset_path=args.goal_ft_template.format(test_split_name),
                                save_path=args.goal_res_template.format(model, test_split_name),
                                metric=constr_strict_eval_goal_metric,
                                example_select=selector)
        elif mode == "plan":
            fewshot_evaluation(model=model, 
                                example_path=args.plan_ft_template.format(example_split_name),
                                dataset_path=args.plan_ft_template.format(test_split_name),
                                save_path=args.plan_res_template.format(model, test_split_name),
                                metric=constr_validate_plan_metric,
                                example_select=selector)
        elif mode == "list":
            fewshot_evaluation(model=model, 
                                example_path=args.list_ft_template.format(example_split_name),
                                dataset_path=args.list_ft_template.format(test_split_name),
                                save_path=args.list_res_template.format(model, test_split_name),
                                metric=constr_eval_list_goal_metric,
                                example_select=selector)
        elif mode == "lang":
            fewshot_evaluation(model=model, 
                                example_path=args.lang_ft_template.format(example_split_name),
                                dataset_path=args.lang_ft_template.format(test_split_name),
                                save_path=args.lang_res_template.format(model, test_split_name),
                                metric=constr_eval_lang_goal_metric,
                                example_select=selector)
        elif mode == "check":
            for checker in args.domain_checker_list:
                complete_test_split = args.checker_name_template.format(dataset=test_split_name, checker=checker)
                fewshot_evaluation(model=model, 
                                    example_path=args.goal_ft_template.format(example_split_name), # use normal examples
                                    dataset_path=args.check_ft_template.format(complete_test_split),
                                    save_path=args.check_res_template.format(model, complete_test_split),
                                    metric=constr_eval_check_metric,
                                    example_select=selector)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--mode", required=True, choices=["plan", "goal", "list", "lang", "check"])
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--example", type=str)
    parser.add_argument("--model", type=str, default="code-davinci-002", choices=["code-davinci-002", "text-davinci-003"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    random.seed(10)
    args = parse_args()
    openai.api_key = args.api_key
    assert("ROOT_DIR" in os.environ)
    root_dir = os.environ["ROOT_DIR"]
    with open(args.config) as f_conf: 
        config = yaml.safe_load(f_conf)
        args.checker_name_template = config["checker_name_template"]
        args.domain_checker_list = config["domain_checker_list"] 

        # TODO isn't there a better way to construct these paths?
        args.goal_ft_template = os.path.join(root_dir, config["goal_finetuning_template"])
        args.plan_ft_template = os.path.join(root_dir, config["plan_finetuning_template"])
        args.list_ft_template = os.path.join(root_dir, config["list_finetuning_template"])
        args.lang_ft_template = os.path.join(root_dir, config["lang_finetuning_template"])
        args.check_ft_template = os.path.join(root_dir, config["check_finetuning_template"])
        
        args.goal_res_template = os.path.join(root_dir, config["goal_result_template"])
        args.plan_res_template = os.path.join(root_dir, config["plan_result_template"])
        args.list_res_template = os.path.join(root_dir, config["list_result_template"])
        args.lang_res_template = os.path.join(root_dir, config["lang_result_template"])
        args.check_res_template = os.path.join(root_dir, config["check_result_template"])
        
        fewshot_test(args, config[args.task_type], args.mode, args.example)