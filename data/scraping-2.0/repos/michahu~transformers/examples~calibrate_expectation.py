from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

from generate_with_calibration import get_lookahead_entropies
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def calibrate(model, tokenizer, path, save_path, vocab_size, batch_size=512, alpha=0.0, top_k=0, iters=10, threshold=1e-5, device='cpu'):
    alpha = torch.tensor([alpha], requires_grad=True)
    total_loss = CEL(model, tokenizer, path, alpha, vocab_size, batch_size, top_k, device)
    print(f'Total loss: {total_loss.item()}. Alpha: {alpha.item()}')
    last_alpha = alpha.item()
    
    for _ in range(iters):
        grad_a = torch.autograd.grad(total_loss, alpha, create_graph=True)
        grad2_a = torch.autograd.grad(grad_a, alpha)
        alpha.data -= (grad_a[0] / grad2_a[0]).data
        np.savez(save_path, alpha=alpha.item())
        
        total_loss = CEL(model, tokenizer, path, alpha, vocab_size, batch_size, top_k, device)
        print(f'Total loss: {total_loss.item()}. Alpha: {alpha.item()}')
        
        if abs(alpha.data - last_alpha) < threshold:
            break
            
        last_alpha = alpha.item()
    
    return alpha

def CEL(model, tokenizer, path, alpha, vocab_size, batch_size=512, top_k=0, device='cpu'):   
    # calculates the CEL on a single context.
    def CELHelper(context):
        N = len(context)
        context_CEL = torch.tensor([0.0])

        for i in range(1, N):
            with torch.no_grad():
                context_i = torch.tensor(context[:i], dtype = torch.long, device=device).unsqueeze(0)

                inputs = {'input_ids': context_i}
                next_logits = model(**inputs)[0][:, -1, :].detach().cpu()

                if top_k == 0:
                    candidates = None  
                else:
                    candidates = torch.argsort(next_logits[0], descending=True,)[:top_k]
                
                lookahead_ents = get_lookahead_entropies(
                    model = model,
                    context = context_i[0],
                    batch_size = batch_size,
                    vocab_size = vocab_size,
                    candidates = candidates,
                    device = device
                ).cpu()

                next_probs = F.softmax(next_logits, dim=-1)[0]
                
                if top_k != 0:
                    # replace uncomputed entropies with average (for centered adjustment)
                    next_probs = next_probs[lookahead_ents != -1]
                    top_average_ent = (lookahead_ents[lookahead_ents != -1] * next_probs / next_probs.sum()).sum()
                    lookahead_ents[lookahead_ents != -1] = top_average_ent
            
            # context[i] is the next word
            context_CEL -= torch.log(
                F.softmax(next_logits - alpha * lookahead_ents, dim=-1)[0][context[i]]
            )
        return context_CEL
    
    total_CEL = torch.tensor([0.0])

    with open(path) as fp:
        for line in fp:
            context = tokenizer.encode(line)
            # one way to fix memory issues: uncomment the below
            # if (len(context) > 100):
            #    continue
            total_CEL += CELHelper(context)
            
    return total_CEL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--training_path", default=None, type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--save_path", default=None, type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    print(args)

    vocab_size = tokenizer.vocab_size
    print('vocab_size:', vocab_size)

    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)

    alpha = calibrate(
        model=model,
        tokenizer=tokenizer,
        path=args.training_path,
        save_path=args.save_path,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        alpha=args.alpha,
        top_k=args.top_k,
        device=args.device,
    )
    print(alpha)

if __name__ == '__main__':
    main()
