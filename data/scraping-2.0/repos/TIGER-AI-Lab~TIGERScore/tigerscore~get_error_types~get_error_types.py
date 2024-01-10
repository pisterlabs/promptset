# Example usage
"""
This file isn't used in final version.
"""
import os
import sys
import fire
import json
from pathlib import Path
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_API_BASE"] = ""
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
sys.path.append("../")
from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
from xgptscore.constants import EVAL_ASPECTS
from string import Template

TEMPLATE = """\
You are evaluating an ${task} task. Some errors in an incorrect output could be attributed to the following aspects:
${aspects_descriptions}

Please elaborate 10 specific error types for each aspect above. Each error type should represent a specific error that falls under the aspect. Error types should be mutually exclusive and collectively exhaustive.\
"""


def main(
    task: str,
):
    
    task_aspects = EVAL_ASPECTS[task]
    prompt = Template(TEMPLATE).substitute(
        task=task,
        aspects_descriptions="\n".join([f"- {aspect}: {description}" for aspect, description in task_aspects.items()])
    )
    prompts = [prompt]
    chatmls = [[{"role": "system",
                 "content": " You are an AI assistant that helps people find information."},
                {"role": "user",
                 "content": prompt}] for prompt in prompts[:1]]

    chatml_prompts = [_chatml_to_prompt(chatml) for chatml in chatmls]
    results = openai_completions(chatml_prompts, model_name="gpt-4")
    output_file = Path("./error_types/" + task + ".txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results['propmts'] = prompts
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)