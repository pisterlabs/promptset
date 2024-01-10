import torch
from torch import nn
import openai

class bot(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        openai.api_key =  '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'

    def make_response(self, first_inputs):
        

        with torch.no_grad():
            sentences = []
            # output_sentences = [tokenizer.encode(x, add_prefix_space=True) for x in output_sentences_string]
            # prompt = [tokenizer.encode(x, add_prefix_space=True) for x in first_input_string]
            for i in range(len(first_inputs)):
                
                #total_string  = "There is office in the following response:" + output_sentences_string[i]
               # total_string  = "Make the following response full of office:" + output_sentences_string[i]
                total_string = first_inputs[i]
                sentences.append(f"Context: {total_string}\nResponse:")
            reply_string = []

            # start_sequence = "\nPerson 1:"
            # restart_sequence = "\nPerson 2: "

            response = openai.Completion.create(
                engine="ada",
                prompt=sentences,
                temperature=0,
                max_tokens=40,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=["\n"]
                )
            for i in range(len(sentences)):

                reply_string.append(response['choices'][i]['text'])
        # print(reply_string)
        # print("response=",reply_string)
            # reply_string = tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i].strip()]

        return reply_string
