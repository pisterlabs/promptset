import json
from pathlib import Path
from typing import Any

import openai
from tqdm import tqdm

from common import (
    calculate_cost,
    get_key,
    get_result,
    init_argparser,
    log_args,
    logger,
    make_chat_request,
    make_msg,
)

CONTRADICTION_SENTENCE_PROMPTS = [
    "Generate a sentence that contradicts the following:",  # 0
]


def gen_contradiction_sentence(
    model: str, sentence: str, prompt: str
) -> dict[str, Any]:
    response = make_chat_request(
        model=model,
        messages=[
            make_msg(
                "system",
                "You are a helpful assistant that generates contradictions.",
            ),
            make_msg("user", prompt),
            make_msg("user", sentence),
        ],
    )

    return response


def run_contradiction_sentence(
    model: str,
    prompt: str,
    input_path: Path,
    output_path: Path | None,
) -> None:
    data = json.loads(input_path.read_text())
    inputs = [i["sentence2"] for i in data if i["label"] == "ENTAILMENT"]
    responses = [
        gen_contradiction_sentence(model, entailment, prompt)
        for entailment in tqdm(inputs)
    ]
    output = [get_result(response) for response in responses]

    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    cost = sum(calculate_cost(model, response) for response in responses)
    print(f"\nCost: ${cost}")


def main() -> None:
    parser = init_argparser()
    args = parser.parse_args()
    log_args(args, args.args_path)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.prompt < 0 or args.prompt >= len(CONTRADICTION_SENTENCE_PROMPTS):
        raise IndexError(
            f"Invalid prompt index: {args.prompt}. Choose one between 0 and"
            f" {len(CONTRADICTION_SENTENCE_PROMPTS)})"
        )
    run_contradiction_sentence(
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        prompt=CONTRADICTION_SENTENCE_PROMPTS[args.prompt],
    )


if __name__ == "__main__":
    if not hasattr(__builtins__, "__IPYTHON__"):
        main()
