import argparse
import os
from typing import Optional
import yaml

from openai import OpenAI
from tqdm import tqdm
from gpt_gleam.chat import ChatContextCreator, chat, print_messages
from gpt_gleam.configuration import ChatCompletionConfig

from gpt_gleam.data import Stance, iterate_post_frame_labeled_pairs
from gpt_gleam.predictions import JsonlPredictionsWriter
from gpt_gleam.progress import ChatCompletionProgress


def main(
    config: ChatCompletionConfig,
    data_path: str,
    cfact_path: str,
    frame_path: str,
    output_path: str,
    total: Optional[int] = None,
    debug: bool = False,
):
    creator = ChatContextCreator(config)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=os.getenv("OPENAI_TIMEOUT", 90),
    )

    if total is None:
        print("Counting total number of examples (requires iteration)...")
        total = sum(
            1
            for _ in iterate_post_frame_labeled_pairs(
                data_path, frame_path, skip_stances=[Stance.Not_Relevant], cfact_path=cfact_path
            )
        )
        print(f"Total predictions: {total:,}")

    with JsonlPredictionsWriter(output_path) as preds, ChatCompletionProgress(
        total=total, seen=len(preds), disable=debug
    ) as bar:
        for post, frame, _ in iterate_post_frame_labeled_pairs(
            data_path, frame_path, skip_stances=[Stance.Not_Relevant], cfact_path=cfact_path
        ):
            ex_id = f"{post.id}-{frame.id}"
            if ex_id in preds:
                continue
            if post.cfacts is None or frame.id not in post.cfacts:
                print(f"Skipping example due to missing cfacts: {post.id}, {frame.id}")
                continue
            f_cfacts = post.cfacts[frame.id]
            if len(f_cfacts) != 3:
                print(
                    f"Skipping example due to missing stance value cfacts: {post.id}, {frame.id} only has {len(f_cfacts)} cfacts"
                )
                continue

            accept, reject, no_stance = f_cfacts
            prompt_kwargs = {
                "accept_rationale": accept.rationale,
                "reject_rationale": reject.rationale,
                "no_stance_rationale": no_stance.rationale,
            }

            messages = creator.create_context(post, frame, **prompt_kwargs)
            completion = chat(
                client,
                delay=config.delay,
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=config.seed,
            )
            if completion is None:
                print(f"Skipping example due to API safety error: {post.id}, {frame.id}")
                continue
            content = completion.choices[0].message.content
            preds.add({"id": ex_id, "post_id": post.id, "f_id": frame.id, "content": content})
            messages.append({"role": "assistant", "content": content})
            if debug:
                print_messages(messages)
            bar.update(completion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="path to data jsonl file")
    parser.add_argument("--cfact_path", type=str, help="path to cfact jsonl file")
    parser.add_argument("--frame_path", type=str, required=True, help="path to frames json file")
    parser.add_argument("--output_path", type=str, required=True, help="path to output jsonl file")
    parser.add_argument("--total", type=int, help="total number of examples to process")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = ChatCompletionConfig(**config)
    main(
        config=config,
        data_path=args.data_path,
        cfact_path=args.cfact_path,
        frame_path=args.frame_path,
        output_path=args.output_path,
        total=args.total,
        debug=args.debug,
    )
