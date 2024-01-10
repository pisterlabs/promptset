from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import LogitsProcessorList
import transformers

import json
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import openai
import re

from utils import setup_openai, normalised_ngrams, compute_cosine_similarity, save_dataset_to_disk
from steer import STEER

### GENERATION CONFIGS ###

# Greedy
greedy_config = GenerationConfig(
    do_sample=True,
    num_beams=1,
    temperature=0.5,
    num_return_sequences=1,
)

# Beam search
beam_config = GenerationConfig(
    num_beams=3,
    early_stopping=True,
    temperature=0.94,
    num_return_sequences=1,
    do_sample=True,
)

# Top-k
top_k_config = GenerationConfig(
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    temperature=0.85,
)

# Top-p nucleus sampling
nucleus_config = GenerationConfig(
    do_sample=True,
    top_p=0.92,
    top_k=0,
    temperature=1.0,
)

# Contrastive search
contrastive_config = GenerationConfig(
    do_sample=True,
    penalty_alpha=0.35, 
    top_k=10,
    temperature=1.0,
)

config_dictionary = {
    "greedy": greedy_config,
    "beam": beam_config,
    "top_k": top_k_config,
    "nucleus": nucleus_config,
    "contrastive": contrastive_config,
}

@dataclass
class Experiment:
    task: str
    method: str
    model: str

class PipelineBase:

    def __init__(self, experiment: Experiment):
        self.task = experiment.task
        self.method = experiment.method
        self.model = experiment.model
        self._set_attributes()
        self.local_model_path = f"/g/data/y89/cn1951/{self.model}-{self.task}-tiny"
        self.parent_model_path = f"/g/data/y89/cn1951/{self.model}"
        self.synthetic_dataset_path = f"../results/datasets/{self.task}-{self.model}-{self.method}.txt"
        self.real_dataset_path = f"../data/{self.task}.json"
        self.output_dir = "../results/datasets/"
    
    def _set_attributes(self):
        if self.task == "hypotheses":
            self.prompt = "### Instruction: Generate a scientific hypothesis about astronomy in the style of an Arxiv paper.\n ### Hypothesis:"
            self.split = "Hypothesis"
            self.max_length = 100
        elif self.task == "comments":
            self.prompt = "### Instruction: Generate a non-toxic social media comment.\n ### Comment:"
            self.split = "Comment"
            self.max_length = 75
        elif self.task == "commonsense":
            self.prompt = "### Instruction: Generate a multiple-choice question that relies on common-sense to answer.\n ### Multiple-choice question:"
            self.split = "Multiple-choice question"
            self.max_length = 75
        else:
            pass

class InferenceTracker(PipelineBase):

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self._set_embeddings()
        self.synthetic_dataset = []
        self.synthetic_embeddings = []
        self.metrics = {
            'normalised_ngrams': [],
            'cosine_similarity': [],
        }
        self.deployment_name = setup_openai()

    def _set_embeddings(self):
        self.real_embeddings = np.load(f"../results/embeddings/{self.task}-{self.model}-{self.method}-real.npy")

    def add_synthetic_example(self, example: str):
        self.synthetic_dataset.append(example)
        self.synthetic_embeddings.append(self.embed_example(example))
        self.metrics['normalised_ngrams'].append(self.normalised_ngrams(n=3))
        self.metrics['cosine_similarity'].append(self.cosine_similarity())

    def embed_example(self, example: str):
        response = openai.Embedding.create(
            engine=self.deployment_name,
            input=example,
        )
        return response['data'][0]['embedding']
    
    def cosine_similarity(self, centroid: bool = False) -> float:
        return compute_cosine_similarity(self.synthetic_embeddings, self.real_embeddings, centroid)

    def normalised_ngrams(self, n = 3) -> float:
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        return normalised_ngrams(self.synthetic_dataset, self.real_dataset, tokenizer, n)

class InferencePipeline(PipelineBase):

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        #self.inf_tracker = InferenceTracker(experiment)

    def create_pipeline(self):
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=torch.bfloat16, 
                                                                device_map="auto", trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.parent_model_path)
        pipeline = transformers.pipeline(
            "text-generation",
            model=fine_tuned_model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        return pipeline, tokenizer
    
    def generate_response(self, pipeline, tokenizer):

        sequences = pipeline(
            self.prompt,
            max_length=self.max_length,
            generation_config=config_dictionary[self.method],
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        return self.format_example(sequences[0]['generated_text'], batch=False)
    
    def format_batch(self, examples_list):
        formatted_examples = []
        for examples in examples_list:
            for example in examples:
                text = self.format_example(example, batch=True)
                formatted_examples.append(text)
        return formatted_examples
    
    def format_example(self, example, batch=False):
        # Split the generated text on '### Hypothesis:' and take the second part
        if batch: text = example['generated_text'].split(f'{self.split}:')[1].strip()
        else: text = example.split(f'{self.split}:')[1].strip()
        if self.task == "hypotheses":
            # Remove excess question marks, replace them with just one
            text = text.rstrip('?') + '?'
        elif self.task == "commonsense":
            # Remove everything after the E. Question ... i.e. remove F. -> onwards if it exists
            text = text.split("F.")[0]
            text = text.split(">")[0]
            pattern = r"(Answer: [A-E]\.?)(.*$)"
            # Replace the matched pattern with the first group only (i.e., the first 'Answer: X.').
            cleaned_line = re.sub(pattern, r'\1', text)
            # Ensure the cleaned line ends with a period.
            if not cleaned_line.endswith('.'):
                cleaned_line += '.'
            text = cleaned_line
        else:
            # Remove all newlines, put each example on one line
            text = text.replace("\n", " ")
            # Put a "###" at the start and end of text
            text = "### " + text + " ###"
        return text
    
    def generate_synthetic_dataset(self, num_examples: int = 100, save_to_disk: bool = False, batch_size: int = 8):
        """
        NOTE: batch_size = 32 seems to perform the fastest.
        """
        dataset = []
        pipeline, tokenizer = self.create_pipeline()

        if batch_size == 0:
            # Just do it sequentially
            for _ in tqdm(range(num_examples)):
                example = self.generate_response(pipeline, tokenizer)
                #self.inf_tracker.add_synthetic_example(example)
                dataset.append(example)

        else:
            gen_dataset = GenerationDataset(num_examples=num_examples, prompt=self.prompt)
            if self.task == "hypotheses":
                pipeline.tokenizer.pad_token_id = 42 # EOS token is '?'
                end_token = 42
            else:
                pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id
                end_token = tokenizer.eos_token_id
            for out in tqdm(pipeline(gen_dataset, max_length=self.max_length, generation_config=config_dictionary[self.method],
                                     num_return_sequences=1, eos_token_id=end_token, pad_token_id=end_token,
                                     batch_size=batch_size), total=len(gen_dataset)):
                dataset.append(out)
            dataset = self.format_batch(dataset)

        if save_to_disk:
            save_dataset_to_disk(self.output_dir, dataset, self.task, self.model, self.method)
        
        return dataset
    
class GenerationDataset(Dataset):

    def __init__(self, num_examples, prompt):
        self.num_examples = num_examples
        self.prompt = prompt

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        return self.prompt
    
class STEERPipeline(InferencePipeline):

    def __init__(self, experiment: Experiment, gamma=0.2, eta=0.6, num_neg_prompts=10):
        super().__init__(experiment)
        self.gamma = gamma
        self.eta = eta
        self.num_neg_prompts = num_neg_prompts
        self.base_model_path = self.parent_model_path
        with open(self.real_dataset_path, "r") as file:
            self.running_examples = list([example['text'].split(f"{self.split}: ")[-1] for example in json.load(file)])

    def create_pipeline(self):
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(self.local_model_path, torch_dtype=torch.bfloat16, 
                                                                device_map="auto", trust_remote_code=True, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=torch.bfloat16, 
                                                          device_map="auto", trust_remote_code=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.parent_model_path)
        
        return base_model, fine_tuned_model, tokenizer

    def generate_negative_prompt(self, tokenizer):
        neg_examples = random.sample(self.running_examples, self.num_neg_prompts)
        neg_examples = ' '.join(neg_examples)
        return tokenizer(neg_examples, return_tensors='pt')['input_ids'].to('cuda')

    def generate_response(self, base_model, fine_tuned_model, tokenizer, eta, gamma):
        if eta == -1 and gamma == -1:
            eta = self.eta
            gamma = self.gamma
        neg_prompt = self.generate_negative_prompt(tokenizer)

        prompt_tensor = tokenizer(self.prompt, return_tensors='pt').to('cuda')

        if self.task == "hypotheses": end_token = 42
        else: end_token = tokenizer.eos_token_id
        
        sequences = fine_tuned_model.generate(
            input_ids=prompt_tensor['input_ids'],
            attention_mask=prompt_tensor['attention_mask'],
            max_new_tokens=self.max_length,
            eos_token_id=end_token,
            pad_token_id = tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([
                STEER(gamma=gamma, eta=eta, base_model=base_model, fine_tuned_model=fine_tuned_model, 
                      uncond=neg_prompt),
            ]),
            renormalize_logits=True,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
        )

        decoded_sequence = self.format_example(tokenizer.decode(sequences[0]), batch=False)
        self.running_examples.append(decoded_sequence)

        return decoded_sequence
    
    def generate_synthetic_dataset(self, num_examples: int = 16, save_to_disk: bool = False, batch_size: int = 8,
                                   eta=-1, gamma=-1):
        """
        NOTE: batch_size = 32 seems to perform the fastest.
        """
        dataset = []
        base_model, fine_tuned_model, tokenizer = self.create_pipeline()

        if batch_size == 0:
            for _ in tqdm(range(num_examples)):
                example = self.generate_response(base_model, fine_tuned_model, tokenizer, eta, gamma)
                #self.inf_tracker.add_synthetic_example(example)
                dataset.append(example)

        if save_to_disk:
            save_dataset_to_disk(self.output_dir, dataset, self.task, self.model, self.method)
        
        return dataset


### QUICK RUNS ###

def create_dataset(experiment: Experiment, num_examples: int, save_to_disk: bool = True, batch_size: int = 16):
    # Create the pipeline
    pipeline = InferencePipeline(experiment=experiment)
    # Generate the synthetic dataset
    dataset = pipeline.generate_synthetic_dataset(num_examples=num_examples, save_to_disk=save_to_disk, batch_size=batch_size)
    return dataset

if __name__ == "__main__":
    experiment = Experiment(task="comments", method="nucleus", model="falcon-7b")
    dataset = create_dataset(experiment, num_examples=300, save_to_disk=True, batch_size=32)
    print(dataset)

    # experiment= Experiment(task="comments", method="steer", model="falcon-7b")
    # steer_pipe = STEERPipeline(experiment=experiment, gamma=0.2, eta=0.2, num_neg_prompts=10)
    # dataset = steer_pipe.generate_synthetic_dataset(num_examples=100, batch_size=0, gamma=0.2, eta=0.2, save_to_disk=True)
    # print(dataset)
