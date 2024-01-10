import os
import argparse
import openai


def main():
    # Check that OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY env var must be set: export OPENAI_API_KEY=YOUR_KEY_HERE")

    parser = argparse.ArgumentParser(description="Fine tunes a model for both participants from messenger")
    parser.add_argument("model", type=str, help="The model to use")
    parser.add_argument("--content", type=str, help="The question to ask", required=True)
    parser.add_argument("--name", type=str, help="The name of the person", required=True)
    args = parser.parse_args()

    messages = [
        {"role": "system", "content": f"Your name is {args.name}"},
        {"role": "user", "content": args.content},
    ]
    # model_id "ft:gpt-3.5-turbo:my-org:custom_suffix:id"
    completion = openai.ChatCompletion.create(
        model=args.model,
        messages=messages,
    )
    print(completion.choices[0].message)


if __name__ == "__main__":
    main()
