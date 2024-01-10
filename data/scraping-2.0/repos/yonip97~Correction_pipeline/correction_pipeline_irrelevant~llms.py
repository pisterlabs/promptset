import openai
import tiktoken
from correction_pipeline.utils import collate_fn
from torch.utils.data import DataLoader


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

    def estimate_dataset(self, prompt, dataset):
        dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
        total_cost = 0
        for x in dataloader:
            _, original_text, generated_text, label = x
            original_text, generated_text, label = original_text[0], generated_text[0], label[0]
            model_input = prompt
            model_input += 'original_text: ' + '\n' + original_text + '\n'
            model_input += "generated_text: " + '\n' + generated_text + '\n'
            model_input += 'revised_text: ' + '\n'
            total_cost += self.estimate_cost(model_input, generated_text)
        return total_cost


class LLM_model():
    def __init__(self, prompt_path=None, model='chatgpt-turbo-3.5', API_KEY=None, **kwargs):
        openai.api_key = API_KEY
        if prompt_path != None:
            with open(prompt_path, "r") as file:
                prompt = file.read()
        else:
            prompt = ''
        self.prompt = prompt
        self.model = model

    def get_chatgpt_response(self, input, max_length, **kwargs):
        try:
            message = [{
                "role": "user",
                "content": input,
            }]
            response = openai.ChatCompletion.create(
                engine=self.model,
                messages=message,
                temperature=0,
                max_tokens=max_length
            )
            return response['choices'][0]['message']['content']
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            return None
