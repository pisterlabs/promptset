import torch
import random
from sentence_transformers import SentenceTransformer, util

from transformers import (
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5Tokenizer,
)
from typing import Any
from torch.utils.data.dataset import Dataset
import numpy as np
import json

import openai

openai.api_key = "YOUR_OPENAI_KEY_HERE"
gpt_engine = "text-davinci-002"


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data_pairs: Any):
        originals = []
        labels = []
        for item in data_pairs:
            if len(item) != 2:
                print("Critical Error: input data ill-formatted.")
                continue
            originals.append(item[0])
            labels.append(item[1])

        self.inputs = tokenizer.batch_encode_plus(originals, pad_to_max_length=True)
        self.labels = tokenizer.batch_encode_plus(labels, pad_to_max_length=True)

        assert len(self.inputs["input_ids"]) == len(data_pairs)

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, i):
        source_ids = self.inputs["input_ids"][i]
        target_ids = self.labels["input_ids"][i]
        src_mask = self.inputs["attention_mask"][i]
        target_mask = self.labels["attention_mask"][i]
        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids, "decoder_attention_mask": target_mask}


class FactsGenerator:

    def __init__(self):
        self.device = "cuda:0"
        self.facts_model = T5ForConditionalGeneration.from_pretrained("CogComp/l2d-decomp").to(self.device)
        self.facts_model.eval()
        self.paraphrase_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.num_facts = 3
        self.num_per_candidates = 5
        self.num_trials = 5
        self.eos_id = 1
        self.random_mode = True

        self.openai_correction_cache = json.load(open("open_ai_correction_cache.json"))
        self.all_openai_correction_cache_sentences = []
        for sentence in self.openai_correction_cache:
            self.all_openai_correction_cache_sentences.append(sentence)

        # Open AI
        self.use_openai_correction = True

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def gen_gpt_correction(self, gen_fact):
        if gen_fact in self.openai_correction_cache:
            return self.openai_correction_cache[gen_fact]
        response = openai.Completion.create(
            engine=gpt_engine,
            prompt="Fix the input sentence with correct facts if there are factual errors. \n\nWrong: Mount Fuji is in China. \nCorrect: Mount Fuji is in Japan.\n\nWrong: Amy Winehouse was diagnosed with stage 4 breast cancer in May 2017.\nCorrect: Amy Winehouse was not diagnosed with cancer.\n\nWrong: Barack Obama was born in California on September 4, 1965. \nCorrect: Barack Obama was born in Hawaii on August 4, 1961.\n\nWrong: Ten gallons of seawater weigh 650 pounds.\nCorrect: Ten gallons of seawater weigh approximately 83 pounds.\n\nWrong: Buffalo wings contain capsaicin.\nCorrect: Buffalo wings contain capsaicin. \n\nWrong: The Albany in Georgia has over 50,000 people.\nCorrect: The Albany in Georgia has over 73,000 people.\n\nWrong: {}\nCorrect:".format(gen_fact),
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            best_of=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        corrected = response["choices"][0]["text"].replace("\n", "").strip()
        self.openai_correction_cache[gen_fact] = corrected
        return corrected

    def find_best_match(self, current_facts, generated_facts, seen_generations):
        print("--------Looking for best generations-----------", flush=True)
        print("INFO: Current generations: {}".format(str(current_facts)), flush=True)
        print("INFO: Considerations: {}".format(str(generated_facts)), flush=True)
        current_fact_encodes = None
        if len(current_facts) > 0:
            current_fact_encodes = self.paraphrase_model.encode(current_facts + list(seen_generations), convert_to_tensor=True, device=self.device)
        probs = []
        eos_count = 0
        for d, s in generated_facts:
            if d.startswith("End of"):
                eos_count += 1
            probs.append(s)
        if eos_count >= 4:
            return None
        for i in range(0, len(generated_facts)):
            if self.random_mode:
                selected = random.choices(generated_facts, weights=probs, k=1)[0]
            else:
                selected = generated_facts[i]
            if selected[0].startswith("End of"):
                continue
            if len(current_facts) == 0:
                # if selected[0] not in seen_generations:
                if True:
                    print("INFO: Selected '{}'".format(selected), flush=True)
                    return selected
            else:
                sim_phrase = selected[0]
                if self.use_openai_correction:
                    sim_phrase = self.gen_gpt_correction(sim_phrase)
                gen_fact_encode = self.paraphrase_model.encode([sim_phrase], convert_to_tensor=True, device=self.device)
                sim_scores = util.cos_sim(gen_fact_encode, current_fact_encodes)
                do_not_select = False
                for score in sim_scores[0]:
                    if score >= 0.9:
                        do_not_select = True
                if selected[0] in seen_generations or sim_phrase in seen_generations:
                    do_not_select = True
                if not do_not_select:
                    print("INFO: Selected '{}'".format(selected), flush=True)
                    return selected
        return None

    def gen_facts(self, question):
        all_trial_results = []
        for trail_num in range(0, self.num_trials):
            current_facts = []
            seen_generations = set()
            for i in range(0, self.num_facts):
                print("INFO: Run question '{}', trail_num {}, fact index {}".format(question, str(trail_num), str(i)), flush=True)
                input_seq = question + " </s> Decompositions: " + " </s> ".join(current_facts)
                dataset = LineByLineTextDataset(self.tokenizer, [(input_seq, "nothing here to see")])
                with torch.no_grad():
                    outputs = self.facts_model.generate(
                        input_ids=torch.tensor([dataset[0]["input_ids"]], dtype=torch.long).to(self.device),
                        attention_mask=torch.tensor([dataset[0]["attention_mask"]], dtype=torch.long).to(self.device),
                        num_beams=self.num_per_candidates, num_return_sequences=self.num_per_candidates,
                        num_beam_groups=self.num_per_candidates, diversity_penalty=0.8,
                        max_length=64, forced_eos_token_id=self.eos_id,
                        return_dict_in_generate=True, output_scores=True,
                    )
                    sequences = outputs.sequences
                    sequence_scores = outputs.sequences_scores
                    ret_items = []
                    for sid, sequence in enumerate(sequences):
                        seq_score = float(sequence_scores[sid])
                        ret_items.append([self.tokenizer.decode(sequence, skip_special_tokens=True), seq_score])
                    all_scores = []
                    if len(ret_items) > 0:
                        for _, s in ret_items:
                            all_scores.append(s)
                        all_scores = self.softmax(all_scores)
                        for j in range(0, len(ret_items)):
                            ret_items[j][1] = float(all_scores[j])
                    best_match = self.find_best_match(current_facts, ret_items, seen_generations)
                    if best_match is None:
                        break
                    best_match_truth = None
                    if self.use_openai_correction:
                        best_match_truth = self.gen_gpt_correction(best_match[0])
                        print('GPT Correction: "{}" to "{}"'.format(best_match[0], best_match_truth))
                    applied_generation = best_match[0]
                    if best_match_truth is not None:
                        applied_generation = best_match_truth
                    seen_generations.add(best_match[0])
                    seen_generations.add(applied_generation)
                    print("INFO: Found best match '{}'".format(applied_generation), flush=True)
                    current_facts.append(applied_generation)
            all_trial_results.append(current_facts)
        return all_trial_results


# @input_file: raw questions, formatted as data/strategyqa/dev_qonly.txt
# @output_file: an output json file with generated decompositions
def generate_decomposition(input_file, output_file):
    generator = FactsGenerator()
    lines = [x.strip().split("\t") for x in open(input_file).readlines()]
    f_out = open(output_file, "w")
    for question, answer in lines:
        question = question
        facts = generator.gen_facts(question)
        f_out.write(json.dumps(facts) + "\n")
        f_out.flush()


# @question_file: raw questions, formatted as data/strategyqa/dev_qonly.txt
# @decomp_file: outputs from generate_decomposition(). It should contain the same amount of lines.
# @output_file: an output path
# @limit: how many decomposition sentences to consider
def format_to_entailment_model(question_file, decomp_file, output_file, limit=3):
    decomp_lines = [x.strip() for x in open(decomp_file).readlines()]
    lines = [x.strip() for x in open(question_file).readlines()][:len(decomp_lines)]
    f_out = open(output_file, "w")
    for i, line in enumerate(lines):
        decomp_obj = json.loads(decomp_lines[i])
        for decomp in decomp_obj:
            question = line.split("\t")[0] + " </s> Decomposition: " + " ; ".join(decomp[:limit])
            f_out.write("{}\t{}\n".format(question, line.split("\t")[1]))
