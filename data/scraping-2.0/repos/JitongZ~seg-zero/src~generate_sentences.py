import torch
import argparse
import os
import openai
from transformers import AutoTokenizer, T5ForConditionalGeneration
from itertools import chain
from tqdm import tqdm


class SentenceGenerator:
    def __init__(self, source, target, output_folder) -> None:
        # Path to the file
        self.source_file_path = os.path.join(output_folder, f"{source}.txt")
        self.target_file_path = os.path.join(output_folder, f"{target}.txt")
        self.source = source
        self.target = target
        self.output_folder = output_folder

        # Check if the folder exists, if not, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def generate(self):
        raise NotImplementedError

    def _prompt(self):
        raise NotImplementedError

    def _write_out(self, source_captions, target_captions):
        # Write the sentences to the file, each on a new line
        with open(self.source_file_path, "w") as file:
            text = "\n".join(source_captions)
            text = text.replace("\n\n", "\n")
            file.write(text)
        print(f"File {self.source_file_path} has been successfully saved")

        with open(self.target_file_path, "w") as file:
            for sentence in target_captions:
                file.write(sentence + "\n")
        print(f"File {self.target_file_path} has been successfully saved")

    # TODO: add filter to delete sentences that doesn't contain the source/target words
    def _filter_poor_sentences(self):
        raise NotImplementedError


class T5Generator(SentenceGenerator):
    def __init__(self, source, target, output_folder) -> None:
        super().__init__(source, target, output_folder)

    def _prompt(self):
        source_prompt = f"Provide a caption for images containing a {self.source}. "
        "The captions should be in English and should be no longer than 150 characters."

        target_prompt = f"Provide a caption for images containing a {self.target}. "
        "The captions should be in English and should be no longer than 150 characters."

        return (source_prompt, target_prompt)

    def generate_captions(self, input_prompt, num_sentences):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16
        )
        input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

        res = []
        for i in tqdm(range(num_sentences // 16)):
            outputs = model.generate(
                input_ids,
                temperature=0.8,
                num_return_sequences=16,
                do_sample=True,
                max_new_tokens=128,
                top_k=10,
            )
            res.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return res

    def generate(self, num_sentences):
        source_prompt, target_prompt = self._prompt()

        source_captions = self.generate_captions(source_prompt, num_sentences)
        target_captions = self.generate_captions(target_prompt, num_sentences)

        self._write_out(source_captions, target_captions)


class GPT4Generator(SentenceGenerator):
    def __init__(self, source, target, output_folder, sentences_per_query=20) -> None:
        super().__init__(source, target, output_folder)
        self.sentences_per_query = 20

    def _prompt(self):
        source_prompt = f"You are a piction captioner. Generate {self.sentences_per_query} short sentences describing {self.source}, focusing on physical visual aspects and excluding anything that is non-physical. the sentence must contain the word {self.source}. Separate them by newlines. Do not number the sentences."
        target_prompt = f"You are a piction captioner. Generate {self.sentences_per_query} short sentences describing {self.target}, focusing on physical visual aspects and excluding anything that is non-physical. the sentence must contain the word {self.target}. Separate them by newlines. Do not number the sentences."

        return (source_prompt, target_prompt)

    def _extract_captions(self, completion):
        pass

    def generate_captions(self, input_prompt, num_sentences):
        captions = []
        for i in tqdm(range(num_sentences // self.sentences_per_query)):
            completion = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "user", "content": input_prompt}]
            )
            captions.append(completion.choices[0].message.content.split("\n"))
        return list(chain(*captions))

    def generate(self, num_sentences):
        source_prompt, target_prompt = self._prompt()

        source_captions = self.generate_captions(source_prompt, num_sentences)
        target_captions = self.generate_captions(target_prompt, num_sentences)

        self._write_out(source_captions, target_captions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="black eyes")
    parser.add_argument("--target", type=str, default="blue eyes")
    parser.add_argument("--num_sentences", type=int, default=1000)
    parser.add_argument("--output_folder", type=str, default="assets/sentences/")
    parser.add_argument("--backend", type=str, default="t5")
    args = parser.parse_args()

    if args.backend == "gpt4":
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        gen = GPT4Generator(args.source, args.target, args.output_folder)
    elif args.backend == "t5":
        gen = T5Generator(args.source, args.target, args.output_folder)

    gen.generate(args.num_sentences)
