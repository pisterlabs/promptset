# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model


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
    if not args.SEQ2SEQ:
        SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
    else:
        SPECIAL_TOKENS = ["<s>", "</s>", "madeupword0000", "madeupword0001", "<pad>"]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    token_score = None
    if args.inference == "nucleus":
        for i in range(args.max_length):
            if not args.SEQ2SEQ:
                instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False, SEQ2SEQ=args.SEQ2SEQ)

                input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

                logits = model(input_ids, token_type_ids=token_type_ids)
            else:
                instance = build_input_from_segments(personality, history, current_output, tokenizer, SEQ2SEQ=args.SEQ2SEQ)

                #seq2seq
                input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                target_ids = torch.tensor(instance["target_ids"], device=args.device).unsqueeze(0)
                input_type_ids = torch.tensor(instance["input_type_ids"], device=args.device).unsqueeze(0)
                target_type_ids = torch.tensor(instance["target_type_ids"], device=args.device).unsqueeze(0)

                logits = model(input_ids, input_ids!=tokenizer.convert_tokens_to_ids("<pad>"), target_ids, target_ids!=tokenizer.convert_tokens_to_ids("<pad>"), 
                    return_dict=False,
                )
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :]
 
            logits = logits / args.temperature
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
            
    elif args.inference == "greedy":
        for i in range(args.max_length):
            if not args.SEQ2SEQ:
                instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False, SEQ2SEQ=args.SEQ2SEQ)

                input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

                logits = model(input_ids, token_type_ids=token_type_ids)
            else:
                instance = build_input_from_segments(personality, history, current_output, tokenizer, SEQ2SEQ=args.SEQ2SEQ)

                #seq2seq
                input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                target_ids = torch.tensor(instance["target_ids"], device=args.device).unsqueeze(0)
                input_type_ids = torch.tensor(instance["input_type_ids"], device=args.device).unsqueeze(0)
                target_type_ids = torch.tensor(instance["target_type_ids"], device=args.device).unsqueeze(0)

                logits = model(input_ids, input_ids!=tokenizer.convert_tokens_to_ids("<pad>"), target_ids, target_ids!=tokenizer.convert_tokens_to_ids("<pad>"), 
                    return_dict=False,
                )
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]

            logits = logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
    
            prev = torch.topk(probs, 1)[1] 
            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

    else:
        """ Beam Search using the encoder inputs contained in `batch`.
        """
        beam_size = args.beam_size
        device = args.device
        batch_size = args.batchsize

        # Tile states and memory beam_size times.
        if not args.SEQ2SEQ:
            instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False, SEQ2SEQ=args.SEQ2SEQ)
            input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0) #bsz * seq_len
            token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0) #bsz * seq_len

            input_ids = input_ids.expand(batch_size * beam_size, -1).to(args.device)
            token_type_ids = token_type_ids.expand(batch_size * beam_size, -1).to(args.device)

        else:
            instance = build_input_from_segments(personality, history, current_output, tokenizer, SEQ2SEQ=args.SEQ2SEQ)

            #seq2seq
            input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0) #bsz * seq_len
            target_ids = torch.tensor(instance["target_ids"], device=args.device).unsqueeze(0) #bsz * seq_len
            input_type_ids = torch.tensor(instance["input_type_ids"], device=args.device).unsqueeze(0) #bsz * seq_len
            target_type_ids = torch.tensor(instance["target_type_ids"], device=args.device).unsqueeze(0) #bsz * seq_len

            input_ids = input_ids.expand(batch_size * beam_size, -1).to(args.device)
            target_ids = target_ids.expand(batch_size * beam_size, -1).to(args.device)
            input_type_ids = input_type_ids.expand(batch_size * beam_size, -1).to(args.device)            
            target_type_ids = target_type_ids.expand(batch_size * beam_size, -1).to(args.device)            

        current_output = [current_output * beam_size]
        
        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device
        )
        alive_seq = torch.full(
            [batch_size * beam_size, 1], special_tokens_ids[0], dtype=torch.long, device=device
        )

        alive_score = torch.full(
            [batch_size * beam_size, 1], 0, dtype=torch.long, device=device
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_size - 1), device=device
        ).repeat(batch_size)
        
        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(args.max_length):
            # Generator forward.
            if not args.SEQ2SEQ:
                score = model(input_ids, token_type_ids=token_type_ids)
            else:
                #seq2seq
                score = model(input_ids, input_ids!=tokenizer.convert_tokens_to_ids("<pad>"), target_ids, target_ids!=tokenizer.convert_tokens_to_ids("<pad>"), 
                    return_dict=False,
                )
            if isinstance(score, tuple):  # for gpt2 and maybe others
                score = score[0]

            score = score[:, -1, :] 
            score = score.view(batch_size, beam_size, -1)
            for j, logit in enumerate(score):
                logit = logit[0] / args.temperature
                score[j][0] = top_filtering(logit, top_k=args.top_k, top_p=args.top_p)
           
            log_probs = F.log_softmax(score, dim=-1).view(batch_size * beam_size, -1)  
            vocab_size = log_probs.size(-1)

            if step < args.min_length:
                for s in special_tokens_ids:
                    log_probs[:, s] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = args.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = tokenizer.decode(words, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=(args.eval_type != 'f1'))
                        if len(words) <= 3:
                            continue
                        trigrams = [
                            (words[i - 1], words[i], words[i + 1])
                            for i in range(1, len(words) - 1)
                        ]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty 

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size, rounding_mode='trunc')
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = topk_beam_index + beam_offset[
                : topk_beam_index.size(0)
            ].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )

            is_finished = topk_ids.eq(special_tokens_ids[1])
            if step + 1 == args.max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )
            # Reorder states.
            #current_output = [pred.tolist() for pred in preds]
            current_output =[cand[1:].tolist() for cand in alive_seq]

            if step !=  args.max_length - 1:
                if not args.SEQ2SEQ:
                    input_ids = []
                    token_type_ids = []
                    for c in current_output:
                        instance = build_input_from_segments(personality, history, c, tokenizer, with_eos=False, SEQ2SEQ=args.SEQ2SEQ)
                        input_ids.append(instance["input_ids"])
                        token_type_ids.append(instance["token_type_ids"])                        
                    input_ids = torch.tensor(input_ids, device=args.device)#.unsqueeze(0)
                    token_type_ids = torch.tensor(token_type_ids, device=args.device)#.unsqueeze(0)
                else:
                    input_ids = []
                    target_ids = []
                    for c in current_output:
                        instance = build_input_from_segments(personality, history, c, tokenizer, SEQ2SEQ=args.SEQ2SEQ)
                        input_ids.append(instance["input_ids"])
                        target_ids.append(instance["target_ids"])                        
                    input_ids = torch.tensor(input_ids, device=args.device)#.unsqueeze(0)
                    target_ids = torch.tensor(target_ids, device=args.device)#.unsqueeze(0)
        current_output = results["predictions"][0][0].tolist()[:-1]
    return current_output

def run():
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
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

    #zhangying
    parser.add_argument("--SEQ2SEQ", action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":
    run()
