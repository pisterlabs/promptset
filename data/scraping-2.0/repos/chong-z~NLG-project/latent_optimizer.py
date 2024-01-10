import math
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch
import string

class Semantic_Loss():

    def __init__(self):

        # Load pre-trained model (weights)
        self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # Get perplexity score for this sentence
    def get_perplexity(self, sentence):

        # Remove eos tags
        sentence = sentence.replace("<eos>", "")
        # Remove puncutation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))

        tokenize_input = self.tokenizer.tokenize(sentence)
        # print(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])

        # If the sentence is small (< X), then we have no loss
        loss = 0
        if len(sentence.split()) > 8:
            loss= self.model(tensor_input, lm_labels=tensor_input)

            


        return math.exp(loss)
