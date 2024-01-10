import openai

from io_utils import read_jsonlines, read_json, write_jsonlines
from datasets import Dataset, concatenate_datasets

openai.api_key = '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'

def gpt_attack(example, no_wm_attack=False):
    # assert attack_prompt, "Prompt must be provided for GPT attack"
    # attack_prompt="As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
    
    # attack_prompt="paraphrase the following paragraphs:\n"

    attack_prompt="You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n"

    gen_row = example

    if no_wm_attack:
        original_text = gen_row["no_bl_output"]
    else:
        original_text = gen_row["w_bl_output"]

    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}
    # query_msg = [{'role': 'system', 'content': 'You are a helpful assistant.'},
    # {'role': 'user', 'content': attack_prompt},
    # {'role': 'assistant', 'content': 'No problem.'},
    # {'role': 'user', 'content': original_text}]

    from tenacity import retry, stop_after_attempt, wait_random_exponential

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens):
        return openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    outputs = completion_with_backoff(
        model='gpt-3.5-turbo',
        messages=[query_msg],
        temperature=0.7,
        max_tokens=1000,
    )

    attacked_text = outputs.choices[0].message.content
    print(original_text)
    print("check-------------------")
    print(attacked_text)
    # assert (
    #     len(outputs.choices) == 1
    # ), "OpenAI API returned more than one response, unexpected for length inference of the output"
    example["w_bl_num_tokens_generated"] = outputs.usage.completion_tokens
    example["w_bl_output"] = attacked_text
    # print(outputs.usage.completion_tokens)
    # print(len(attacked_text.split(" ")))
    # print(len(original_text.split(" ")))
    # print(example["w_bl_num_tokens_generated"])
    # if args.verbose:
    #     print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
    #     print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")

    return example

def str_replace_bug_check(example,idx):
    baseline_before = example["baseline_completion"]
    example["baseline_completion"] = baseline_before.replace(example["truncated_input"][:-1],"")
    if example["baseline_completion"] != baseline_before:
        # print("baseline input replacement bug occurred, skipping row!")
        return False
    else:
        return True

run_base_dir = f"/egr/research-dselab/renjie3/renjie/LLM/watermark_LLM/lm-watermarking/experiments/results/all_runs_07131520"
meta_name = "gen_table_meta.json"
gen_name = "gen_table_w_metrics.jsonl"
gen_table_meta_path = f"{run_base_dir}/{meta_name}"
gen_table_path = f"{run_base_dir}/{gen_name}"
attack_path = f"{run_base_dir}/gpt_attacked_100.jsonl"

# load the raw files
gen_table_meta = read_json(gen_table_meta_path)
print(gen_table_meta)
gen_table_lst = [ex for ex in read_jsonlines(gen_table_path)]
# print(gen_table_lst)
gen_table_ds = Dataset.from_list(gen_table_lst[:100])

print(gen_table_ds)

print(f"Original dataset length={len(gen_table_ds)}")
gen_table_ds_filtered = gen_table_ds.filter(str_replace_bug_check,batched=False,with_indices=True)
print(f"gen_table_ds_filtered length={len(gen_table_ds_filtered)}")

# gen_table_ds_filtered[0] = 
# gpt_attack(gen_table_ds_filtered[0])
attacked_ds = gen_table_ds_filtered.map(gpt_attack)
# print(gen_table_ds_filtered[0]["real_completion_length"])

attacked_ds_lst = [ex for ex in attacked_ds]
write_jsonlines(attacked_ds_lst, attack_path)
