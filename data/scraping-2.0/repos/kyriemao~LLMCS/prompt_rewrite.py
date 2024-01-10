from IPython import embed
import os
import json
import time
import argparse
from tqdm import tqdm, trange
from promptor import RewritePrompter
from generator import OpenAIGenerator, OPENAI_KEYS
from utils import set_seed, get_finished_sample_ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demonstration_file_path", type=str, required=True)
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True, help='output rewrite path.')
    parser.add_argument("--n_completion", type=int, required=True, help='the number of completions for generation')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--open_ai_key_id", type=int, choices=[0,1,2,3,4,5], required=True)
    
    
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "parameters.txt"), "w") as f:
        params = vars(args)
        f.write(json.dumps(params, indent=4))
        
    return args


def main():
    args = get_args()    
    set_seed(args) 
    
    # model and promptor setting
    promptor = RewritePrompter()
    model_kwargs = {"n": args.n_completion, "top_p": 1, "temperature": 0.7, "max_tokens": 64, "logprobs": 1, "stop": promptor.stop_tokens}
    openai_key = OPENAI_KEYS[args.open_ai_key_id]
    generator = OpenAIGenerator(args.model_name, openai_key, **model_kwargs)
    
    # demos
    with open(args.demonstration_file_path, "r") as f:
        demo_dialogs = json.load(f)
    demo_prompt = promptor.build_demo_prompt(demo_dialogs)
    
    # test_dataset    
    output_file_path = os.path.join(args.work_dir, "rewrites.jsonl")
    finished_samples = get_finished_sample_ids(output_file_path)
    with open(args.test_file_path, "r") as f:
        test_dialogs = json.load(f)
    begin_time = time.time()
    
    # predict
    with open(output_file_path, "a+") as f:
        for i in trange(len(test_dialogs)):
            dialog = test_dialogs[i]
            conv_id = dialog['conv_id'] 
            turns = dialog['turns']
            
            pre_prompt = demo_prompt
            last_predicted_rewrite, last_response = None, None
            for j in trange(len(turns)):
                turn_id = turns[j]['turn_id']
                sample_id = "{}_{}".format(conv_id, turn_id)
                
                # prompting
                prompt = promptor.build_this_turn_prompt_for_prediction(pre_prompt, turns[j], last_predicted_rewrite, last_response)
                
                # generating
                if sample_id in finished_samples:
                    rewrite_list = finished_samples[sample_id]['predicted_rewrite']
                elif int(turn_id) == 1:
                    rewrite_list = [turns[j]['question']] * args.n_completion
                else:
                    while True:
                        try:
                            rewrite_list = generator.generate(prompt, promptor.parse_returned_text)
                        except ValueError as e:
                            print("{}, re-generating...".format(e.args[0]))
                            continue
                        break

                
                # update last info, only use the top-1 to construct the conversation when generating n completions
                last_predicted_rewrite = rewrite_list[0]
                last_response = turns[j]['response']
                pre_prompt = prompt
                
                # recording
                record = {}
                record['sample_id'] = sample_id
                record['predicted_rewrite'] = rewrite_list
                if sample_id not in finished_samples:
                    f.write(json.dumps(record))
                    f.write('\n')

                # output info
                print("{}, sample_id: {} \n predicted rewrites: {}".format(args.work_dir, sample_id, rewrite_list))

                
    print("{} Generation ok!, time cost {}".format(args.work_dir, time.time() - begin_time))


if __name__ == '__main__':
    main()
