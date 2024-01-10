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

from src.few_shot.utils import save_to_disk

# from src.few_shot.generation import LLaMA

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


# Set the seed
seed = 0
torch.manual_seed(seed)


def get_llama(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model.seqlen = 2048
    return model


from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import torch
import os
import jsonlines
import argparse
from tqdm import tqdm
from datasets import load_dataset

BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"
# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


if __name__ == "__main__":

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

    current_files = []

    c = 0
    # prompt = "Given the following question and context, provide an answer " \
    #          "that best fits the question. Ensure that the answer " \
    #          "is a span in the context."
    prompt = (
        "Answer the question based on the context below. If the question cannot be answered "
        "using the information provided, then answer with 'I don't know'."
    )

    # to test
    # prompt = "Your task is to answer a question based on the given context. If the information provided " \
    #          "is insufficient to answer the question, please respond with 'I don't know.' Your response " \
    #          "should be clear and concise, providing only relevant information necessary to answer the question. " \
    #          "Please note that you should make every effort to provide an accurate and complete response based " \
    #          "on the available information."

    model_name = "llama-13b-hf"
    # model_name = "google/flan-ul2"
    model_identifier = model_name.split("/")[-1]

    save_path = (
        BASE_PATH
        + f"src/data/squad/few_shot_{model_identifier}_qa_eval_seed_{args.seed}"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_llama(model_name).to("cuda")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # all_files = os.listdir(BASE_PATH + f"src/data/squad/few_shot_{model_identifier}_qg_temp_0.7/")
    # files = [file for file in all_files if file not in current_files]
    # print(files)
    # files = ["llama_collated_data_with_answers_processed_context_irrelevance.jsonl"]
    files = ["counterfactual_data_llama_13b_v1_qg_pipeline_all_data_cleaned.jsonl"]

    for file in files:

        # if c<=9:
        #     continue
        print(f"Processing file: {file}")
        examples = []
        with jsonlines.open(BASE_PATH + f"src/data/squad/" + file) as reader:
            # with jsonlines.open(BASE_PATH + f"src/data/squad/few_shot_{model_identifier}_qg_temp_0.7/" + file) as reader:
            for example in tqdm(reader):
                c += 1
                id = example["id"].split("_")[0]
                context = example["context"]
                question = example["question"]
                tokens_to_remove = ["[", "'", '"', "]"]
                # Create a translation table that maps each unwanted token to None
                translator = str.maketrans({token: None for token in tokens_to_remove})
                question = question.translate(translator).strip()

                orig_example = [sample for sample in squad_data if sample["id"] == id][
                    0
                ]

                orig_context = orig_example["context"]
                orig_question = orig_example["question"]
                orig_answer = orig_example["answers"]

                input = (
                    f"{prompt} \nContext: {orig_context} \nQuestion: {orig_question} "
                    f"\nAnswer: {orig_answer['text'][0]}"
                    f"\n{prompt} \nContext: {context} \nQuestion: {question} "
                    f"\nAnswer: "
                )

                inputs = tokenizer(input, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(
                    inputs, max_new_tokens=50, temperature=0.7, top_p=1, top_k=40,
                )
                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # remove the context from the output
                output = output[len(input) :]
                answer = output.split("\n")[0]
                result = {
                    "id": example["id"],
                    "question": question,
                    "context": context,
                    "answer": answer,
                }
                print("question: ", question)
                print("context: ", context)
                print("answer: ", result["answer"])
                print("original answer: ", example["answers"])
                print("-" * 10)
                examples.append(result)

                if c == 10:
                    break

        # save_to_disk(examples, f"{save_path}/counterfactual_samples_{model_identifier}_{c}.jsonl")
        # c += 1
