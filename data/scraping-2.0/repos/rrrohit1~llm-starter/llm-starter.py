import langchain
from langchain import HuggingFaceHub
import os
import argparse

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "TOKEN"
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl")


def main(prompt):
    # prompt = "Alice has a parrot. What animal is Alice's pet?"
    completion = llm(prompt)
    print(f"The prompt is: {prompt}")
    print(f"The answer is: {completion}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        action="store",
        type=str,
        help="display the answer of the prompt",
        dest="prompt",
        nargs="+",
        default=["Alice has a parrot. What animal is Alices pet?"],
    )
    args = parser.parse_args()
    main(" ".join(args.prompt))
