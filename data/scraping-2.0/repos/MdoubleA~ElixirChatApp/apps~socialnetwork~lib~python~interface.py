# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Source repository: https://github.com/huggingface/transfer-learning-conv-ai
import torch
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
import torch.nn.functional as F
import warnings
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, cached_path
import random
from os import path
import tarfile


# Constant from Hugging Faces' source code.
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"


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


def sample_sequence(personality, history, tokenizer, model, device, temperature, top_k, top_p, no_sample,
                    max_length, min_length, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
        if i < min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

# -- The following code are my modifications, wrappers, and additions for this project-------
# ------------------------------------------------------------------------------

# Modifies Hugging Faces download_pretrained_model() that's found in utils.py.
# Is renamed and uses a permanent directory rather than a temporary one.
def get_pretrained_model():
	model_path = ".\\lib\\python\\trained_model"

	if not path.exists(model_path):
	    """ Download and extract finetuned model from S3 """
	    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)

	    #logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
	    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
	        archive.extractall(model_path)

	return model_path


# Set hyperparameters
no_sample = True  # Set to use greedy decoding instead of sampling
max_out_utter = 20  # Maximum length of the output utterances
min_out_utter = 1  # Minimum length of the output utterances
seed = 0  # test what happens when using random number.
temperature = 0.7  # Sampling softmax temperature
top_k = 0  # Filter top-k tokens before sampling (<=0: no filtering)
top_p = 0.9  # Nucleus filtering (top-p) before sampling (<=0.0: no filtering)
max_history = 2

# build model and tokenizer

if seed != 0:  # why is this here?
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

#model_checkpoint = ".\\tmpa0_h0vzt"
model_checkpoint = get_pretrained_model()
host_device = "cpu"
# host_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_class, model_class = (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
model = model_class.from_pretrained(model_checkpoint)
model.to(host_device)
add_special_tokens_(model, tokenizer)


convert = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
mk_personality = lambda persona: list(map(convert, persona))


# Persona is a list of strings. history an empty list or the history returned from a previous call to respond.
# Raw_statement is a question or response to a bot in the form as a single string.
def respond(persona, history, raw_statement):
    # Map Elixir types to python types.
    # Elixir data cannot be mutated so make copies of data where necessary, ie converting the list 'history' to a list;
    # that makes a copy of the data which maintains elixir data integrity.
    persona = [x.decode('utf-8') for x in persona]  # Convert list of elixir bit strings to list of python strings.
    raw_statement = raw_statement.decode('UTF-8')  # Elixir bit string to string.
    history = list(history)  # Copy elixir list.
    personality = mk_personality(persona)
    ret_response, return_history = None, None  # Initialize return values.

    if raw_statement:
        history.append(tokenizer.encode(raw_statement))
        with torch.no_grad():
            #print(personality)  # at this point it was been encoded into numbers, additional tokens and all.
            out_ids = sample_sequence(personality, history, tokenizer, model, host_device, temperature, top_k, top_p,
                                      no_sample,
                                      max_out_utter, min_out_utter)
        history.append(out_ids)
        return_history = history[-(2 * max_history + 1):]
        ret_response = tokenizer.decode(out_ids, skip_special_tokens=True)

    return ret_response, return_history


def test_respond():
    persona = ["I'm the worlds most decorated olympian.",
               "The Arizona State Sun Devils are my favorite team. I volunteer there.",
               "I like to party; no else likes me partying. But I like to party."]
    convo = ["hello! how are you?", "i'm good. what do you like to do for fun?"]

    history = []
    for i in range(len(convo)):
        response, history = respond(persona, history, convo[i])
        print(response, history)
