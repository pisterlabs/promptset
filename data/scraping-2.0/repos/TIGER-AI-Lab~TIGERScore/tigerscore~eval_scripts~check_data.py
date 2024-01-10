import sys
sys.path.append("..")
from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
import fire
import json
import random
from string import Template


template = """
${instruction}
${input}

Model-generated output:
${output}

An error analysis provided:
${error_analysis}

Is the error analysis reasonable? Answer me "yes" or "no" only.\
"""

def main(input_file, output_file, model_name="gpt-4", num_samples=None, num_procs=5):
    with open(input_file, "r") as f:
        if input_file.endswith(".jsonl"):
            input_data = [json.loads(line) for line in f]
        elif input_file.endswith(".json"):
            input_data = json.load(f)
    if num_samples is None:
        num_samples = len(input_data)
    print(num_samples)
    input_data = input_data[:num_samples]
    
    def process_data(item):
        prompt = Template(template=template).substitute(
            instruction=item["instruction"],
            input=item["input_context"],
            output=item["hypo_output"],
            error_analysis=item["errors"]
        )
        message = [{
            "role": "user",
            "content": prompt
        }]
        chatml_prompt = _chatml_to_prompt(message)
        return chatml_prompt
        
    prompts = list(map(process_data, input_data))
    print(prompts[0])
    completions = openai_completions(prompts, model_name=model_name, num_procs=num_procs, use_cache=False)
    print(f"Finished generating {len(completions['completions'])} completions.")
    print(f"Total prices: {sum(completions['price_per_example'])}")
    for i, completion in enumerate(completions['completions']):
        input_data[i]["completion"] = completion
    with open(output_file, "w") as f:
        if output_file.endswith(".jsonl"):
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
        elif output_file.endswith(".json"):
            json.dump(input_data, f)
        
if __name__ == "__main__":
    fire.Fire(main)
    