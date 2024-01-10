import openai
import os
import argparse
import tqdm

TOKEN_USAGE = 0


def print_usage(model):
    # Define model costs per 1000 tokens
    model_costs = {
        "ada": 0.0016,
        "babbage": 0.0024,
        "curie": 0.0120,
        "gpt-3.5-turbo": 0.002,
    }

    # Calculate total cost
    total_cost = TOKEN_USAGE * model_costs[model] / 1000
    # Display total cost and tokens used
    print("Tokens used:", TOKEN_USAGE)
    print("Total cost: $", total_cost, sep="")


# Getting top n completions from a finetuned model
# returns a list of strings
def get_top_n_completions(model, prompt, n=1, separator="\n\n###\n\n", stop="||##||"):
    global TOKEN_USAGE
    if model != "gpt-3.5-turbo":
        response = openai.Completion.create(
            prompt=prompt + separator,
            model=model,
            max_tokens=250,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=[stop],
            n=n,
        )
        TOKEN_USAGE += response["usage"]["total_tokens"]
        return [completion["text"].strip() for completion in response["choices"]]
    else:
        completions = []
        setup = "You translate Natural Language prompts to Bash commands. You reply with a single Bash command no matter what. Provide only a single command and no explanations."

        for i in range(n):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": setup},
                    {"role": "user", "content": prompt},
                ],
            )
            TOKEN_USAGE += response["usage"]["total_tokens"]
            completions.append(response["choices"][0]["message"]["content"].strip())

        return completions


def predict_file(model, prompts_file, n, output_file):
    # Read prompts from file
    with open(prompts_file, "r") as f:
        prompts = f.read().split("\n")

    completions = []
    for prompt in tqdm.tqdm(prompts):
        completions.append(get_top_n_completions(model, prompt, n=n))

    # write predictions to the output file
    # Each line looks like this: pred1|||pred2|||pred3|||
    with open(output_file, "w") as f:
        for i, completion in enumerate(completions):
            for version in completion:
                # Escape newline characters
                escaped_version = version.replace("\n", "\\n")
                f.write(escaped_version + "|||")
            f.write("\n")


def main():
    # Authenticating with openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Name of the model, can be ada, curie, or babbage, required, defaults to ada
    parser.add_argument("--model", type=str, required=True)

    # Number of completions per innstance
    parser.add_argument("--n", type=int, default=3)

    # Path to the file containing the prompts, defaults to data/dummy.nl.filtered
    parser.add_argument("--prompts", type=str, default="data/dummy.nl.filtered")

    # Output file path
    parser.add_argument("--output", type=str, required=True)

    # Get arguments
    args = parser.parse_args()

    # Check if the prompts file exists
    if not os.path.exists(args.prompts):
        raise ValueError("Prompts file does not exist:", args.prompts)

    # Set model name
    if args.model == "ada":
        model = "ada:ft-personal:ada-3-epochs-2023-04-18-09-43-53"
    elif args.model == "babbage":
        model = "babbage:ft-personal:babbage-3-epochs-2023-04-18-11-00-05"
    elif args.model == "curie":
        model = "curie:ft-personal:curie-3-epochs-2023-04-18-11-41-28"
    elif args.model == "gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    else:
        raise ValueError("Invalid model name:", args.model)

    # Check if output file directory exists, create if not
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Predict the file and write ooutputs
    predict_file(model, args.prompts, n=args.n, output_file=args.output)

    print_usage(args.model)


if __name__ == "__main__":
    main()
