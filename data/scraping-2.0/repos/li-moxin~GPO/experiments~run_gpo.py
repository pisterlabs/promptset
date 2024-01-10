import random

import fire, datetime, logging

from automatic_prompt_engineer import ape, data, config
from data.instruction_induction.load_data import load_data, tasks, train_sample_index, eval_sample_index, parameters
from evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from evaluation.instruction_induction.exec_accuracy import label_data
import openai

openai.api_key = ''

def run(task_1, task_2, num_subsamples = None, num_demos = None, num_prompts_per_subsample = None, eval_num = None, seed=0, conf_threshold=0.6):
    """
    task_1: source group dataset name, e.g., mnli
    task_2: target group dataset name, e.g., anli
    num_subsamples: (from APE) to split training data into # num_subsample splits
    num_demos: (from APE) each split has # num_demos instances.
    num_prompts_per_subsample: (from APE) for each split, generate #num_prompts_per_subsample prompts (param of LLM, n=num_prompts_per_subsample). it is always set to 1 since we uses LLM temperature=0.
    eval_num: number of validation data.
    seed: 0-4. we select fixed five set of few shot training data for averaged results. change seed to change different training data.
    conf_threshold: confidence threshold.
    num_demos (from APE): not used since we are zero-shot.
    """

    assert task_1 in tasks and task_2 in tasks, 'Task not found!'

    if num_subsamples == None or num_demos == None or num_prompts_per_subsample == None or eval_num == None:
        print("Using default parameters for task", task_1, task_2)
        num_subsamples, num_demos = parameters[task_1]["num_subsamples"], parameters[task_1]["num_demos"]
        num_prompts_per_subsample = 1
        eval_num = num_subsamples * num_demos
        conf_threshold = parameters[task_1]["conf_threshold"]
        print(num_subsamples, num_demos, num_prompts_per_subsample, eval_num, conf_threshold)

    # setting up logging
    filename = "GPO_" + task_1 + "_" + task_2 + "_" + str(num_subsamples) + str(num_demos) + str(num_prompts_per_subsample) + str(eval_num) + str(seed) + "_conf_" + str(conf_threshold) + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print("log saved at", filename)
    # logging.basicConfig(filename = 'results/' + filename, level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('results/' + filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
 
    logging.info("Task 1 {}, Task 2 {}, seed {}".format(task_1, task_2, seed))
    
    induce_data_1, test_data_1 = load_data('induce', task_1), load_data('eval', task_1)
    induce_data_2, test_data_2 = load_data('induce', task_2), load_data('eval', task_2)

    test_data_1 = test_data_1[0][:5], test_data_1[1][:5]
    test_data_2 = test_data_2[0][:5], test_data_2[1][:5]

    logging.info("Creating task 1 data...")
    # Get size of the induce data
    induce_data_size_1 = len(induce_data_1[0])
    prompt_gen_size_1 = min(int(induce_data_size_1 * 0.5), 1000)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data_1 = induce_data_1[0][:prompt_gen_size_1], induce_data_1[1][:prompt_gen_size_1]
    eval_data_1 = induce_data_1[0][prompt_gen_size_1:], induce_data_1[1][prompt_gen_size_1:]
    
    logging.info("Size of train, eval and test data")
    logging.info("{}, {}, {}".format(len(prompt_gen_data_1[0]), len(eval_data_1[0]), len(test_data_1[0])))
    logging.info("Task 1 data sample")
    logging.info(test_data_1[0][:5] + test_data_1[1][:5])

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data_1 = prompt_gen_data_1[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data_1[1]]

    logging.info("Creating task 2 data...")
    # Get size of the induce data
    induce_data_size_2 = len(induce_data_2[0])
    prompt_gen_size_2 = min(int(induce_data_size_2 * 0.5), 1000)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data_2 = induce_data_2[0][:prompt_gen_size_2], induce_data_2[1][:prompt_gen_size_2]
    eval_data_2 = induce_data_2[0][prompt_gen_size_2:], induce_data_2[1][prompt_gen_size_2:]
    
    logging.info("Size of train, eval and test data")
    logging.info("{}, {}, {}".format(len(prompt_gen_data_2[0]), len(eval_data_2[0]), len(test_data_2[0])))
    logging.info("Task 2 data sample")
    logging.info(test_data_2[0][:5] + test_data_2[1][:5])
    
    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data_2 = prompt_gen_data_2[0], [random.sample(output, 1)[0] for output in prompt_gen_data_2[1]]
    
    # Meta Prompt
    prompt_gen_template = "I provide my friend with an instruction. Based on the instruction, I gave him several inputs, and he generated the corresponding outputs. Here are the input-output examples:\n\n[full_DEMO]\n\nPlease briefly illustrate the instruction and describe the output format. The instruction is to [APE]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nOutput: [OUTPUT]"
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    
    logging.info("eval_template: {}".format(eval_template))
    logging.info("prompt_gen_template: {}".format(prompt_gen_template))
    logging.info("demos_template: {}".format(demos_template))
    
    base_config = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': num_subsamples,
            'num_demos': num_demos,
            'num_prompts_per_subsample': num_prompts_per_subsample,
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-0301'
                }
            },
            'replicate_time': 1,
            'seed': train_sample_index[seed],
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task_1,
            'num_samples': min(eval_num, len(eval_data_1[0])),
            'num_few_shot': num_demos,
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-0301'
                }
            },
            'seed': eval_sample_index[seed],
        }
    }
    
    # sampling few shot prompt generation data.
    prompt_gen_data_1 = data.subsample_data(prompt_gen_data_1, num_subsamples * num_demos, conf["generation"])
    prompt_gen_data_2 = data.subsample_data(prompt_gen_data_2, num_subsamples * num_demos, conf["generation"])
    eval_data_1 = data.subsample_data(eval_data_1, num_subsamples * num_demos, conf["evaluation"])
    eval_data_2 = data.subsample_data(eval_data_2, num_subsamples * num_demos, conf["evaluation"])
    assert None not in prompt_gen_data_1[0]
    assert None not in prompt_gen_data_2[0]
    assert None not in eval_data_1[0]
    assert None not in eval_data_2[0]
    
    # Step 1: prompt generation via meta prompt.
    res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                    prompt_gen_data=prompt_gen_data_1,
                                    eval_data=eval_data_1,
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=None,
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)

    logging.info("Finished finding prompts.")
    prompts, scores = res.sorted()
    
    logging.info("Prompts:")
    for prompt, score in list(zip(prompts, scores))[:10]:
        logging.info(f'  {score}: {prompt}')

    
    # Step 2: Prompt Ensemble Labeling Strategy.
    logging.info('Annotating Task 2...')

    annotation_conf = {
        'generation': {
            'num_subsamples': num_subsamples,
            'num_demos': num_demos,
            'num_prompts_per_subsample': num_prompts_per_subsample,
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-0301'
                }
            },
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task_2,
            'num_samples': len(prompt_gen_data_2[0]),
            'num_few_shot': num_demos,
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-0301'
                }
            },
            'ensemble':1,
            'conf_lower_bound': conf_threshold,
        }
    }
    
    annotation_conf = config.update_config(annotation_conf, base_config)
    # Label target data
    prompt_gen_data_2_labeled = label_data(prompts, eval_template, prompt_gen_data_2, demos_template, prompt_gen_data_2, annotation_conf['evaluation'])
    eval_data_2_labeled = label_data(prompts, eval_template, eval_data_2, demos_template, prompt_gen_data_2,  annotation_conf['evaluation'])

    # Step 3: Joint Prompt Optimization.
    logging.info("Regenerating the prompts using mixed data")
    res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                    prompt_gen_data=[prompt_gen_data_1, prompt_gen_data_2_labeled, 'shuffle_all'],
                                    eval_data=[eval_data_1, eval_data_2_labeled, ''],
                                    conf=conf,
                                    base_conf=base_config,
                                    few_shot_data=[prompt_gen_data_1, prompt_gen_data_2_labeled, ''],
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)

    logging.info('Finished finding prompts.')
    prompts, scores = res.sorted()
    logging.info("Prompts:")
    for prompt, score in list(zip(prompts, scores))[:10]:
        logging.info(f'  {score}: {prompt}')

    # Evaluate on test data
    logging.info("Evaluating on test data...")

    test_conf = {
        'generation': {
            'num_subsamples': num_subsamples,
            'num_demos': num_demos,
            'num_prompts_per_subsample': num_prompts_per_subsample,
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-0301'
                }
            },
            'replicate_time': 1,
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task_1,
            'num_samples': len(test_data_1[0]),
            'num_few_shot': num_demos,
            'model': {
                'gpt_config': {
                     'model': 'gpt-3.5-turbo-0301'
                }
            },
        'ensemble': 1,
        }
    }


    # Testing stage.
    logging.info("Run Task 1 test")
    test_res = ape.evaluate_prompts(prompts=prompts,
                                    eval_template=eval_template,
                                    eval_data=test_data_1,
                                    few_shot_data=prompt_gen_data_1,
                                    demos_template=demos_template,
                                    conf=test_conf,
                                    base_conf=base_config)
    test_prompts, test_scores_1 = test_res.in_place()
    
    logging.info("Test Prompts:")
    for prompt, score in list(zip(test_prompts, test_scores_1)):
        logging.info(f'  {score}: {prompt}')

    logging.info("Run Task 2 test")
    test_conf['evaluation']['task'] = task_2
    test_conf['evaluation']['num_samples'] = len(test_data_2[0])
    if prompts[-1] == "Ensemble":
        prompts = prompts[:-1]
    test_res = ape.evaluate_prompts(prompts=prompts,
                                    eval_template=eval_template,
                                    eval_data=test_data_2,
                                    few_shot_data=prompt_gen_data_2,
                                    demos_template=demos_template,
                                    conf=test_conf,
                                    base_conf=base_config)
    test_prompts, test_scores_2 = test_res.in_place()

    logging.info("Test Prompts:")
    for prompt, score in list(zip(test_prompts, test_scores_2)):
        logging.info(f'  {score}: {prompt}')

if __name__ == '__main__':
    fire.Fire(run)
