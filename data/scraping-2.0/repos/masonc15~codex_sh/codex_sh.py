import os
import openai
import argparse
from rich.console import Console


def main():

    console = Console()

    # check if API key is set
    if os.environ.get("OPENAI_API_KEY") is None:
        console.print(
            "Please set your API key in the environment variable OPENAI_API_KEY"
        )
        exit(1)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    # parse arguments
    parser = argparse.ArgumentParser(description="OpenAI Codex CLI cheatsheet")

    # add mandatory argument for input query
    parser.add_argument("query", nargs="+", help="query to search for")

    parser.add_argument("-t",
                        "--temp",
                        default=0.0,
                        type=float,
                        help="Codex model temperature (randomness)")
    parser.add_argument("-n",
                        "--num",
                        default=1,
                        type=int,
                        help="Number of codex predictions to return")
    # add argument for printing prompt file word count
    parser.add_argument(
        "-w",
        "--wordcount",
        action="store_true",
        help="Print word count of prompt file for debugging purposes",
    )
    args = parser.parse_args()

    user_temp = args.temp
    user_num = args.num

    if user_num > 1 and user_temp == 0.0:
        console.print(
            "All completions will be the same with temperature set to 0.0, resulting in wasted tokens.\nChange --num to 1 or set --temp to a value greater than 0.0.",
            style="bold red",
        )
        exit()
    if user_num == 1 and user_temp > 0.0:
        console.print(
            "For best results with a single completion, set temperature to 0.0 with --temp 0.0 or omit the --temp argument.",
            style="bold red",
        )
        exit()

    prompt_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "prompts.txt")

    # optional debugging function
    def get_wordcount(prompt_filepath):
        # get count of individual words in prompt file
        with open(prompt_filepath) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        # remove empty values from list
        content = [x for x in content if x]
        # convert each value of list to number of words in string
        content = [len(x.split()) for x in content]
        # sum up all values in list
        return sum(content)

    if args.wordcount:
        console.print(f"Word count: {get_wordcount(prompt_filepath)}")

    prompt_input = "".join(open(prompt_filepath))
    user_query = " ".join(args.query)

    with console.status(
            "[bold green]Generating response..."
    ) as status:  # show loading message while generating completion
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt_input + user_query + "\nCommand:",
            temperature=user_temp,
            max_tokens=250,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["Query"],
            n=user_num,
        )

        if user_num == 1:
            # print same response but in blue italic
            console.print("[green]" + response["choices"][0]["text"].strip())
        else:
            cheat_answer = "\n".join(
                [choice["text"].strip() for choice in response["choices"]])
            console.print(
                f"Here are [bold red]{user_num}[/bold red] completions with temperature set to [bold red]{user_temp}[/bold red]:"
            )
            # print cheat answers with numbers for each entry
            for i, answer in enumerate(cheat_answer.splitlines()):
                console.print(f"[bold red]{i + 1}[/bold red]: [green]{answer}")
            # exit context manager
            status.stop()


if __name__ == "__main__":
    main()
