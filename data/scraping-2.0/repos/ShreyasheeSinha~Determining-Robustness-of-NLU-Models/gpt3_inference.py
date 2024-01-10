import os
import openai
import hashlib
import pickle
import argparse
import util.load_utils as load_utils
import time
from tqdm import tqdm

class OpenAICommunicator():

    def __init__(self, options):
        data_path = options["data_path"]
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.data = load_utils.load_data(data_path)
        self.template = "Premise: {sentence1}\nHypothesis: {sentence2}\n\nDoes the premise entail the hypothesis?\nAnswer:"
        self.cache_path = options["cache_path"]
        self.save_path = options["save_path"]
        self.cached_responses = self.load_cache_if_exists()

    def load_cache_if_exists(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as handle:
                cache_file = pickle.load(handle)
                return cache_file
        else:
            return {}

    def make_openai_api_call(self, prompt):
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=1,
                top_p=1.0,
                frequency_penalty=0.1,
                presence_penalty=0.0
            )
            return self.parse_api_response(response)
        except openai.error.ServiceUnavailableError:
            print("Service unavailable error hit")
            time.sleep(20)
            return self.make_openai_api_call(prompt)

    def parse_api_response(self, response):
        choices = response["choices"]
        return choices[0]["text"]

    def run_inference(self):
        sentence1_list = self.data["sentence1"]
        sentence2_list = self.data["sentence2"]
        index_list = self.data.index.values.tolist()

        for sentence1, sentence2, index in tqdm(zip(sentence1_list, sentence2_list, index_list), total=len(sentence1_list)):
            prompt = self.template.format(sentence1=sentence1, sentence2=sentence2)
            hashed_prompt = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if hashed_prompt in self.cached_responses:
                response_text = self.cached_responses[hashed_prompt].lower().strip()
            else:
                response_text = self.make_openai_api_call(prompt).lower().strip()
                self.cached_responses[hashed_prompt] = response_text
                with open(self.cache_path, 'wb') as handle:
                    pickle.dump(self.cached_responses, handle)
                time.sleep(5)

            response_to_pred_dict = {"yes": 1, "no": 0}
            if response_text.lower() in response_to_pred_dict:
                prediction = int(response_to_pred_dict[response_text])
                self.data.loc[index, "prediction"] = prediction

        if not os.path.exists(self.save_path):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.data['gold_label'] = self.data['gold_label'].astype(int)
        self.data.to_csv(self.save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the dataset jsonl file", default="./RTE_dev.jsonl")
    parser.add_argument("--cache_path", help="Path with file to save GPT3 responses", default="./gpt3_cache/cache.pickle")
    parser.add_argument("--save_path", help="Path to save model predictions", default="./model_predictions/gpt3/model_predictions_rte_dev.csv")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    options = {}
    options["data_path"] = args.data_path
    options["cache_path"] = args.cache_path
    options["save_path"] = args.save_path

    openai_communicator = OpenAICommunicator(options)
    openai_communicator.run_inference()