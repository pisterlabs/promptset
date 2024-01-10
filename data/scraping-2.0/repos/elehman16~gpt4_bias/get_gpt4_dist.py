import os
import json
import openai
import argparse
from tqdm import tqdm
from utils import run_prompts
from config import PROMPTS_EXPLICIT_DEMOGRAPHICS, PROMPTS_NO_DEMOGRAPHICS

def generate_prompts(condition: str, demographics: bool) -> list[list[str]]:
    """Generate a ton of prompts. If demographics is true, explicitely ask the model to include demographic information."""
    all_prompts = []
    prompts_to_use = PROMPTS_EXPLICIT_DEMOGRAPHICS if demographics else PROMPTS_NO_DEMOGRAPHICS

    for prompt in prompts_to_use:
        query = [
            {"role": "user", "content": prompt.format(condition)},
        ]

        all_prompts.append(query)

    return all_prompts


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_dir', type=str, default='output/')
    argparser.add_argument('--temperature', type=float, default=0.7)
    argparser.add_argument('--max_tokens', type=int, default=100)
    argparser.add_argument('--num_samples', type=int, default=25)
    argparser.add_argument('--condition', type=str, required=True)
    argparser.add_argument('--demographics', action='store_true')
    args = argparser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create prompts
    all_prompts = generate_prompts(demographics=args.demographics, condition=args.condition)

    # Run prompts
    results = run_prompts(all_prompts, args.num_samples, args.temperature, args.max_tokens)

    # Save results
    save_str = 'results_{}_temp_{}_num_samples_{}_max_tokens_{}_condition_{}.json'
    save_str = save_str.format(
        'demographics' if args.demographics else 'no_demographics', 
        args.temperature, 
        args.num_samples, 
        args.max_tokens, 
        args.condition
    )

    with open(os.path.join(args.output_dir, save_str), 'w') as f:
        json.dump(results, f, indent=4)

