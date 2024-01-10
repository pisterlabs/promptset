import openai
import tiktoken
from torch.utils.data import DataLoader
from tqdm import tqdm

class Cost_estimator():
    def __init__(self, model, input_price, output_price):
        """
        :param model: The model for we want the encoding for
        :param input_price: Price for 1k tokens in input
        :param output_price: Price for 1k tokens in output
        """
        self.encoding = tiktoken.encoding_for_model(model)
        self.input_price = input_price
        self.output_price = output_price

    def estimate_cost(self, text, output_estimation):
        input_len = self.estimate_input(text)
        estimated_output_len = self.estimate_output(output_estimation)
        return estimated_output_len / 1000 * self.input_price + input_len / 1000 * self.output_price, input_len, estimated_output_len

    def estimate_input(self, text):
        input_encoding = self.encoding.encode(text)
        return len(input_encoding)

    def estimate_output(self, output_estimation):
        output_encoding = self.encoding.encode(output_estimation)
        return len(output_encoding)

    def estimate_for_revision(self, prompt, texts, summaries):
        total_cost = 0
        total_input = 0
        total_output = 0
        for i in tqdm(range(len(texts))):
            text = texts[i]
            summary = summaries[i]
            model_input = prompt
            model_input += 'original_text: ' + '\n' + text + '\n'
            model_input += "summary: " + '\n' + summary + '\n'
            model_input += 'revised summary: ' + '\n'
            item_cost, item_input_len, item_output_len = self.estimate_cost(model_input, summary)
            total_cost += item_cost
            total_input += item_input_len
            total_output += item_output_len
        return total_cost, total_input, total_output

    def estimate_for_classification(self, prompt, texts, summaries):
        total_cost = 0
        total_input = 0
        total_output = 0
        for i in tqdm(range(len(texts))):
            text = texts[i]
            summary = summaries[i]
            model_input = prompt
            model_input += 'text: ' + '\n' + text + '\n'
            model_input += "summary: " + '\n' + summary + '\n'
            item_cost, item_input_len, item_output_len = self.estimate_cost(model_input, 'yes')
            total_cost += item_cost
            total_input += item_input_len
            total_output += item_output_len
        return total_cost, total_input, total_output
    def estimate_for_summarization(self, prompt, texts, summaries):
        total_cost = 0
        total_input = 0
        total_output = 0
        for i in tqdm(range(len(texts))):
            text = texts[i]
            summary = summaries[i]
            model_input = prompt
            model_input += 'text: ' + '\n' + text + '\n'
            item_cost, item_input_len, item_output_len = self.estimate_cost(model_input, summary)
            total_cost += item_cost
            total_input += item_input_len
            total_output += item_output_len
        return total_cost, total_input, total_output


from datasets import load_dataset
from data.factuality_datasets import TRUE_dataset
dataset = load_dataset('xsum', split='train')
print(len(dataset))
# dataset = TRUE_dataset('data/true_data',['summarization'])
# docs = dataset.df['grounding'].tolist()
# summaries = dataset.df['generated_text'].tolist()
docs = [dataset[i]['document'] for i in range(len(dataset))]
summaries = [dataset[i]['summary'] for i in range(len(dataset))]
estimator = Cost_estimator('gpt-4', 0.03, 0.06)
cost, input_len, output_len = estimator.estimate_for_revision(
    prompt='Please revise the following summary to make it factually consistent with the document. Output only the corrected summary and nothing more.',
    texts=docs, summaries=summaries)
print(cost)
cost, input_len, output_len = estimator.estimate_for_classification(prompt='Please classify if the summary is factually consistent with the document.',
                                                                    texts=docs, summaries=summaries)
print(cost)
cost, input_len, output_len = estimator.estimate_for_summarization(prompt='Please summarize the following document.',
                                                                    texts=docs, summaries=summaries)
print(cost)
# from data.factuality_datasets import TRUE_dataset
# dataset = TRUE_dataset('data/true_data',['summarization'])
# estimator = Cost_estimator('gpt-4',0.03,0.06)
# print(estimator.estimate_dataset('This is a test prompt',dataset))
