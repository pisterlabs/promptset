# Modified by God Bennett for usage in Cryptosynth Aeon Blender3d. See "God_Edit" sections bottom of file.
# This converts original python console dialog, to a setup that simply accepts user input, and a function that captures Ai response, so that
# response can be eventually allocated to a blender3d ui dialog associated with Aeon Ai.
# Huge thanks to original excellent project: https://github.com/huggingface/transfer-learning-conv-ai



import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

#Small Edit by God Bennett/Cryptosynth Labs, to facilitate blender relative import.
#"from train" changed to "from .train"
#"from utils" changed to "from .utils"
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from .train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from .utils import get_dataset, download_pretrained_model

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

#############################################
# God_Edit: Split run function into new class NeuralNetConversationalModule and function getNeuralNetConversationalResponse below.
# This converts original python console dialog, to a setup that simply accepts user input, and a function that captures Ai response, so that
# response can be eventually allocated to a blender3d ui dialog associated with Aeon Ai.



#God_Edit_Addition_Part_A: Add new class NeuralNetConversationalModule, which is taken as a portion of original run () function
#God_Edit: This in short facilitates setup of Ai conversational module, by saving state related to loaded weights as class atttributes,
#          ..usable in getNeuralNetConversationalResponse.
class NeuralNetConversationalModule ():

    def __init__(self):
        #God_Edit: The variables below are now part of new class "NeuralNetConversationalModule".
        self.history = []
        self.args = None
        self.tokenizer = None
        self.model = None
        self.personality = None
    
    def setup (self):
        parser = ArgumentParser()
        parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
        parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
        parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
        parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
        parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

        parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
        parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
        parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
        parser.add_argument("--seed", type=int, default=0, help="Seed")
        parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
        parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
        parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
        self.args = parser.parse_args()


        if self.args.model_checkpoint == "":
            if self.args.model == 'gpt2':
                raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
            else:
                self.args.model_checkpoint = download_pretrained_model()

                
        """
        #GOD_CRUCIAL_EDIT!!: This was removed from code, because dataset loading process, as well as seeing process froze blender ui.
        #This was replaced with a hard-coded personality initializer, where an observed personality is taken from random process, still enabling
        #dynamic Ai responses after this initial personality is chosen.
        #The differences:
        #1. Original method took 2to5 minutes to return answer
        #1b. Original method always enabled a different persona or ai identity to appear
        #2. New method takes about 10 seconds to reeturn answer
        #2.b New method uses only 1 selected persona, and still gives dynamic ai responses as usual
        #2.c Logging also removed from new method.
        if self.args.seed != 0:
            random.seed(args.seed)
            torch.random.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)


        logger.info("Sample a personality")
        dataset = get_dataset(self.tokenizer, self.args.dataset_path, self.args.dataset_cache)
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        self.personality = random.choice(personalities)            
        """
        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if self.args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        self.tokenizer = tokenizer_class.from_pretrained(self.args.model_checkpoint)
        self.model = model_class.from_pretrained(self.args.model_checkpoint)
        self.model.to(self.args.device)
        add_special_tokens_(self.model, self.tokenizer)


        #chosen by God Bennett for Aeon 0019, instead of process which extracts peronslaity from dataset loading, which froze blender3d
        self.personality = [[249, 604, 4183, 498, 4663, 239], [249, 1252, 485, 1074, 1793, 29364, 239], [249, 604, 694, 32700, 488, 651, 239], [249, 256, 258, 481, 4735, 239], [249, 1252, 485, 1074, 246, 11821, 239]]
        

#God_Edit_Addition_Part_B: Add getNeuralNetConversationalResponse, which is taken as a portion of original run () functionB
def getNeuralNetConversationalResponse (userInput,instance_NeuralNetConversationalModule):
    nnc = instance_NeuralNetConversationalModule
    nnc.history.append(nnc.tokenizer.encode(userInput))
    with torch.no_grad():
        out_ids = sample_sequence(nnc.personality, nnc.history, nnc.tokenizer, nnc.model, nnc.args)
    nnc.history.append(out_ids)
    nnc.history = nnc.history[-(2*nnc.args.max_history+1):]
    out_text = nnc.tokenizer.decode(out_ids, skip_special_tokens=True)
    return out_text





#example usage (Call testResponse())

def testResponse ( ):
    userInput = "What is your name?"
    nnc = NeuralNetConversationalModule()
    nnc.setup()
    aiResponse = getNeuralNetConversationalResponse ( userInput, nnc )
    print(aiResponse)

