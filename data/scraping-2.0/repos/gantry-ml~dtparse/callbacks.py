import gantry
import openai
from dataclasses import asdict

class GantryCallback:
    def __init__(self, app_name: str, **tags):
        self.app_name = app_name
        self.tags = tags
        gantry.init()
    
    def __call__(self, prompt: str, args: tuple, kwargs: dict, completion: openai.Completion, llm, model_id):
        inputs = kwargs
        for i, arg in enumerate(args):
            inputs[f"_arg{i}"] = arg
        outputs = {"completion": completion.choices[0].text}
        tags = {
            "prompt": prompt, 
            "openai_id": completion["id"],
            "openai_usage_prompt_tokens": completion["usage"].get("prompt_tokens"),
            "openai_usage_completion_tokens": completion["usage"].get("completion_tokens"),
            "openai_usage_total_tokens": completion["usage"].get("total_tokens"),
            "model_id": model_id,
            **asdict(llm),
            **self.tags,
        }
        # Get around tags needing to be strings
        tags = {k: str(v) for k, v in tags.items()}
        gantry.log_record(
            self.app_name,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
        )
