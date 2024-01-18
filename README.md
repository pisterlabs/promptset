# Prompt Linting

<!-- TODO: Replace link when paper is publicly available -->

Paper: [PromptSet: A Programmer’s Prompting Dataset](https://github.com/pisterlabs/prompt-linter)

## Usage (quick-start):

```python
from datasets import load_dataset

promptset = load_dataset("pisterlabs/promptset")

# iterate all prompts
for prompt_list in promptset["train"]["prompts"]:
  for prompt in prompt_list:
    pass
```

## Organization

1. `data`: contains all the raw data collected from Github.
2. `devGPT`: contains all the processed data collected from DevGPT's Zenodo repository. Check directory for more details.
3. `gen_prompts`: contains code to process and collect prompt data.
4. `analytics`: contains code to analyze the data collected.

## Reproducing Results

1. Download and unzip the repository snapshot as of January 10, 2024. [repos.zip](https://promptset.s3.amazonaws.com/repos.zip)
2. Clone tree-sitter-py `git clone https://github.com/tree-sitter/tree-sitter-python`
3. Run `python -m gen_prompts.find_prompts --run_id 0 --repo_dir {path_to_unzipped_repos} --threads 8`, this parses all the content data to find likely prompt areas.
4. Run `python -m gen_prompts.reader --run_id 0`, here we format and clean the parsed values
5. Run `python -m gen_prompts.upload_ds --run_id 0`, this creates a PR against the pisterlabs/promptset HF repo.
