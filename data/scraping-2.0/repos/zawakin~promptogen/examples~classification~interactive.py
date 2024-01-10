import time
from typing import List

import typer

import promptogen as pg
from examples.base import make_output_path
from examples.classification.dataset_loader import DatasetLoader, IMDbSentimentDataset, TweetEvalEmotionDataset
from examples.llm.openai_util import OpenAITextLLM

app = typer.Typer(add_completion=True)

formatter = pg.KeyValuePromptFormatter()
llm = OpenAITextLLM("gpt-3.5-turbo")
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)


@app.command("tweet_eval_emotion")
def run_tweet_eval_emotion():
    dataset = TweetEvalEmotionDataset(seed=43)
    run_prompt_interactively(dataset)


@app.command("imdb_sentiment")
def run_imdb_sentiment():
    dataset = IMDbSentimentDataset(seed=43)
    run_prompt_interactively(dataset)


def run_prompt_interactively(dataset: DatasetLoader):
    prompt_to_test = pg.Prompt.from_json_file(make_output_path(dataset.attributes.name + "_with_reason.json"))
    input_key = dataset.attributes.input_key
    output_key = dataset.attributes.output_key

    while True:
        # Get user's input from standard input
        user_input = input("Enter your input text: ")
        example_input: pg.Value = {input_key: user_input}

        time.sleep(3)

        # Run the prompt
        try:
            resp = prompt_runner.run_prompt(prompt_to_test, example_input)
        except Exception as e:
            print(f"Error: {e}")
            continue

        # Print the generated text
        print(f"Generated text: {resp[output_key]}")


if __name__ == "__main__":
    print("Initializing...")
    app()
