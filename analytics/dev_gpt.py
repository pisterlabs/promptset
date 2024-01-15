import json

with open("dev_gpt_prompts.json") as f:
    prompts = json.load(f)

with open("dev_gpt_prompts_v2.json", "w") as f:
    json.dump(prompts["Prompts"], f, indent=4, ensure_ascii=False)
