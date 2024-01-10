import os
import jsonlines
from tqdm import tqdm
import torch

from typing import Optional

from datasets import load_dataset
from langchain import PromptTemplate, FewShotPromptTemplate
from transformers import AutoModelForCausalLM, T5Tokenizer, AutoTokenizer, AutoConfig

# from src.few_shot.utils import save_to_disk
from src.few_shot.together import infer

# Set the seed
seed = 0
torch.manual_seed(seed)


BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def save_to_disk(data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(data, total=len(data), desc="Saving samples ... "):
            writer.write(example)


class LLMClient:
    def __init__(
        self,
        template,
        model_name_or_path: str = None,
        tokenizer_name_or_path: Optional[str] = None,
        data_path: str = None,
        threshold: float = 0.5,
        task: str = "qg",
        max_new_tokens: int = 50,
        temperature: float = 0.01,
        top_p: float = 1,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        save_results: bool = True,
        max_samples: int = None,
        stop: str = "\n",
    ):
        self.model = model_name_or_path
        self.tokenizer = tokenizer_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.data_path = data_path
        self.task = task
        self.threshold = threshold
        self.save_results = save_results
        self.max_samples = max_samples
        self.stop = stop.split(";")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.prompt_template = PromptTemplate(
        #     input_variables=["context"] if self.task == "qg" else ["context", "question"],
        #     template=template,
        # )

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map="auto"
        )  # , torch_dtype=torch.bfloat16)
        model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        return model, tokenizer

    def _create_few_shot_prompt(self, examples, context, question=None):
        example_template = """
Context: {context}
Question: {question}
Answer: {answer}
""".strip()

        suffix = """
Context: {context}
Question: 
""".strip()
        example_prompt = PromptTemplate(
            input_variables=["context", "question", "answer"],
            template=example_template,
        )

        if self.task == "qg":
            few_shot_template = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix=qg_template if self.task == "qg" else qa_template,
                suffix=suffix,
                input_variables=["context"],
                example_separator="\n",
            )
            prompt = few_shot_template.format(context=context)
        else:
            few_shot_template = FewShotPromptTemplate(
                examples=examples,
                example_prompt=example_prompt,
                prefix=qa_template,
                suffix=suffix,
                input_variables=["context", "question"],
                example_separator="\n",
            )
            prompt = few_shot_template.format(context=context, question=question)
        return prompt

    def generate(self):
        c = 0
        skipped_instances = 0
        examples = []
        model, tokenizer = self._load_model()
        model_identifier = self.model.split("/")[-1]
        save_path = (
            BASE_PATH
            + f"src/data/squad/t5_squad_counterfactuals/{model_identifier}_{self.task}_pipeline_remaining_temp_0.7"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        file_names = []
        file_path = BASE_PATH + f"src/data/squad/{self.data_path}"
        if self.task == "qg":
            file_names = [file_path]
        elif self.task == "qa":
            all_files = os.listdir(file_path)
            file_names = [
                os.path.join(file_path, file)
                for file in all_files
                if file not in current_files
            ]

        model.eval()
        for file_name in file_names:
            with jsonlines.open(file_name) as reader:
                for example in tqdm(reader):
                    try:
                        id = example["id"].split("_")[0]
                        context = example["context"]
                        orig_example = [
                            sample for sample in squad_data if sample["id"] == id
                        ][0]
                        # print(orig_example)
                        orig_context = orig_example["context"]
                        orig_question = orig_example["question"]
                        orig_answer = orig_example["answers"]
                        c += 1

                        if c <= 73000:
                            continue

                        if self.max_samples:
                            if c == self.max_samples:
                                break
                        if self.task == "qa":
                            # preprocess the question
                            # remove any new line characters
                            question = example["question"]
                            question = question.replace("\n", "")
                            # check for truncated question words
                            # check for truncated question words
                            question_words = [
                                "what",
                                "who",
                                "when",
                                "where",
                                "why",
                                "how",
                                "which",
                                "whom",
                                "whose",
                            ]
                            for word in question_words:
                                if question.lower().startswith(word[1:]):
                                    # replace first word with the correct word
                                    question = word + question[len(word[1:]) :]
                                    # question = word + question[3:]
                                    break

                        prompt = self._create_few_shot_prompt(
                            [
                                {
                                    "context": orig_context,
                                    "question": orig_question,
                                    "answer": orig_answer["text"][0],
                                }
                            ],
                            context,
                            question=question if self.task == "qa" else None,
                        )
                        # print(prompt)

                        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            repetition_penalty=self.repetition_penalty,
                            do_sample=True,
                            # num_return_sequences=1,
                            early_stopping=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # remove the context from the output
                        output = output[len(prompt) :]

                        # output = infer(
                        #     model_name="GPT-JT-6B-v1",
                        #     prompt=prompt,
                        # )

                        result = {"id": example["id"], "context": context}

                        if self.task == "qg":
                            # add the question mark if it is not present
                            # if output[-1] != "?":
                            #     output = output + "?"
                            # question = output

                            # for stop_word in self.stop:
                            #     if stop_word != '' and stop_word in output:
                            #         question = output[:output.find(stop_word)]
                            #     # in some cases the stop word is at the beginning of the output
                            #     if question == "":
                            #         question = output[output.find(stop_word):]
                            # Split the text into lines
                            lines = output.split("\n")

                            # Extract the question and answer
                            question = lines[0].replace("Question: ", "")
                            question = question.strip()
                            answer = lines[1].replace("Answer: ", "")
                            answer = answer.replace("<bot>:", "")
                            answer = answer.strip()

                            result["question"] = question
                            result["answer"] = answer

                        elif self.task == "qa":
                            answer = output
                            # print("Output:", output)
                            for stop_word in self.stop:
                                if stop_word != "" and stop_word in output:
                                    answer = output[output.find(stop_word) :]
                                    # remove the first stop word from answer
                                    answer = answer.split(stop_word)
                                    answer = [a for a in answer if a != ""]
                                    if answer:
                                        answer = answer[0]
                                    else:
                                        answer = "I don't know."

                                # in some cases the stop word is at the beginning of the output
                            #     if answer == "":
                            #         answer = output[output.find(stop_word):]
                            #         answer = answer.replace(stop_word, "")
                            # if answer == "":
                            #     answer = answer.split(stop_word)
                            #     answer = [a for a in answer if a != ""][0]
                            skip_words = [
                                "A: ",
                                "A. ",
                                "Answer: ",
                                "Output: ",
                                "Label: ",
                            ]
                            for skip_word in skip_words:
                                if skip_word in answer:
                                    answer = answer.replace(skip_word, "")

                            # if answer.startswith("A: ") or answer.startswith("A. "):
                            #     answer = answer[3:]
                            # elif answer.startswith("Answer: "):
                            #     answer = answer.replace("Answer: ", "")
                            # question = example["question"]
                            result["question"] = question
                            result["answer"] = answer

                        # print("Context:", context)
                        # print("Question:", question)
                        # print("Answer:", answer)
                        # print("-" * 100)

                        examples.append(result)
                        if self.save_results:
                            if c % self.threshold == 0:
                                save_to_disk(
                                    examples,
                                    f"{save_path}/counterfactual_samples_{model_identifier}_{c}.jsonl",
                                )
                                examples = []
                    except Exception as e:
                        # print(outputs)
                        skipped_instances += 1
                        print(f"Skipped instance {c} due to error: {e}.")
                        continue

            # save the remaining examples
            if self.save_results:
                if examples:
                    save_to_disk(
                        examples,
                        f"{save_path}/counterfactual_samples_{model_identifier}_{c}.jsonl",
                    )
            # break


if __name__ == "__main__":

    qg_template = """
As a question generator, your task is to create a concise and clear question that can be answered by an answer span within a given context. 
The context should be a piece of text, such as a news article or historical document, and the question should require understanding and analysis of the information presented in the context. 
Your generated question should focus on key details or events described in the context and should demonstrate your ability to identify important information. 
Additionally, please ensure that your question is specific enough to have a single correct answer within the given context. 
Please note that you may need to read through the provided context multiple times to fully understand its contents before generating an appropriate question. 
""".strip()

    chatgpt_qg_template = """
As a question generator, your task is to create a clear and concise question that can be answered by an answer span within a given context. 
The context should be a piece of text, such as a news article or historical document, and the question should require understanding and analysis of the information presented in the context. 
Your generated question should focus on key details or events described in the context, requiring readers to carefully read and analyze the provided text. 
Please ensure that your question is specific enough to have only one correct answer span within the given context. 
Please note that you should aim for clarity and concision while still maintaining accuracy and relevance to the provided text.
""".strip()

    qa_template = """
As a language model, your task is to generate an answer that is strictly a span in the given context provided with a specific question.
Please provide a concise and relevant response that directly answers the question asked within the context.
Your answer should be accurate and reflect the information presented in the context.
Please note that you may need to use contextual clues and understanding to identify the most appropriate section of text to use as your answer span.


Context: {context}


Question: {question}


Answer: 
"""

    task = "qg"
    model = "togethercomputer/GPT-JT-6B-v1"
    client = LLMClient(
        template=qg_template if task == "qg" else qa_template,
        model_name_or_path=model,
        tokenizer_name_or_path=model,
        data_path="t5_squad_counterfactuals/rag_counterfactuals_complete_min_filtered_dedup.jsonl",
        threshold=1000,
        task=task,
        max_new_tokens=50,
        temperature=0.7,
        top_p=1,
        top_k=40,
        repetition_penalty=1.0,
        save_results=True,
        max_samples=None,
    )

    client.generate()
