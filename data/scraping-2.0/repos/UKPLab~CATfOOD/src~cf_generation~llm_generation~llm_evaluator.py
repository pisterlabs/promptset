import os
import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from langchain import PromptTemplate
import jsonlines
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, T5Tokenizer, AutoTokenizer, AutoConfig

from src.cf_generation.llm_generation.utils import save_to_disk


BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set the seed
    torch.manual_seed(args.seed)

    # load squad data
    dataset = load_dataset("squad", "plain_text")
    train_data = dataset["train"]
    squad_data = [
        sample
        for sample in tqdm(
            train_data, total=len(train_data), desc="Loading SQuAD data ... "
        )
    ]

    template = """ 
        Given the question: \n
        {query}
        Decide if the following retrieved context is relevant to the {answer}: \n
        {result}
        Answer in the following format: \n
        "Context is relevant: True or False." \n """.strip()

    GRADE_DOCS_PROMPT_FAST = PromptTemplate(
        input_variables=["query", "result", "answer"], template=template
    )

    device = torch.device("cuda:0")
    model_name = "google/flan-ul2"

    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_identifier = model_name.split("/")[-1]
    model_id = "alpaca"
    save_path = BASE_PATH + f"src/data/squad/{model_id}_qa_relevance_seed_{args.seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(
        BASE_PATH,
        f"src/data/squad/counterfactual_data_alpaca_13b_v2_qg_pipeline_all_data_cleaned.jsonl",
    )
    files = [file_path]
    skipped = 0
    c = 0
    for file_name in files:
        examples = []
        with jsonlines.open(file_name) as reader:
            for example in tqdm(reader):
                try:
                    # c+=1
                    # if c <= 25000:
                    #     continue
                    id = example["id"].split("_")[0]
                    context = example["context"]
                    question = example["question"]
                    answer = example["answers"]["text"][0]

                    # print("Given ans:", example["answers"])

                    orig_example = [
                        sample for sample in squad_data if sample["id"] == id
                    ][0]

                    orig_context = orig_example["context"]
                    orig_question = orig_example["question"]
                    orig_answer = orig_example["answers"]

                    input = GRADE_DOCS_PROMPT_FAST.format(
                        query=question, result=context, answer=answer
                    )
                    # print(input)
                    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
                    # outputs = generator.generate([input], max_new_tokens=50, top_p=0.95)

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=10,
                            temperature=0,
                            top_p=1,
                            top_k=40,
                            repetition_penalty=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    c += 1
                    # if c==50:
                    #     break

                    result = {
                        "id": example["id"],
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "context_relevance": output,
                    }
                    # print(result)
                    # break
                    examples.append(result)
                    if c % 5000 == 0:
                        save_to_disk(
                            examples,
                            f"{save_path}counterfactual_samples_{model_id}_{c}.jsonl",
                        )
                        examples = []
                # break
                except Exception as e:
                    print("Skip")
        # save the remaining examples
        if examples:
            save_to_disk(
                examples, f"{save_path}counterfactual_samples_{model_id}_{c}.jsonl"
            )
