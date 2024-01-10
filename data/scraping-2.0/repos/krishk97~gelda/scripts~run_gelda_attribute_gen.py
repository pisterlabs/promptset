import os
import json
import argparse
import openai

from gelda import generate_attributes


def get_arguments():
    parser = argparse.ArgumentParser(description="Attribute generation using chatgpt")
    parser.add_argument(
        "--save_path", "-s",
        type=str,
        default="output/attributes.json",
        help="path to save attributes",
    )
    parser.add_argument(
        "--caption", "-c",
        type=str,
        required=True,
        help="caption to generate attributes for",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="openai model name",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="openai model temperature",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="openai model max tokens",
    )
    parser.add_argument(
        "--n_attempts",
        type=int,
        default=10,
        help="number of tries to generate response from chatgpt",
    )
    parser.add_argument(
        "--n_generations", "-g",
        type=int,
        default=5,
        help="number of times to generate list of attributes and labels",
    )
    parser.add_argument(
        "--n_attrs", "-a",
        type=int,
        default=5,
        help="number of attribute categories to generate",
    )
    parser.add_argument(
        "--n_labels", "-l",
        type=int,
        default=5,
        help="number of attribute labels to generate",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        required=False,
        help="timeout for chatgpt request",
    )
    return parser.parse_args()


def main(args):
    # Generate attributes
    data = generate_attributes(
        args.caption,
        n_attrs=args.n_attrs,
        n_labels=args.n_labels,
        n_generations=args.n_generations,
        n_attempts=args.n_attempts,
        chat_kwargs={"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens}
    )

    # Save data
    save_dir = os.path.dirname(args.save_path)
    save_fname = os.path.basename(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, save_fname), 'w') as f:
        print(f"saving results to: {os.path.join(save_dir, save_fname)}")
        json.dump(data, f, ensure_ascii=False)


if __name__ == "__main__":
    # get arguments
    args = get_arguments()

    # get openai api key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # set timeout
    if args.timeout is not None:
        from gelda.utils import chatgpt_utils
        chatgpt_utils.CHATGPT_TIMEOUT = args.timeout

    # run main
    main(args)
