import typer

import promptogen as pg
from examples.base import make_output_path
from examples.classification.dataset_loader import DatasetLoader, IMDbSentimentDataset, TweetEvalEmotionDataset
from examples.llm.openai_util import OpenAITextLLM
from promptogen.prompt_collection import PromptCreatorPrompt
from promptogen.prompt_tool import PromptWithReasoningTransformer, TextLLMReasoningExtractor

app = typer.Typer()


@app.command("tweet_eval_emotion")
def run_tweet_eval_emotion():
    dataset = TweetEvalEmotionDataset(seed=43)
    setup_prompt_by_dataset(dataset)


@app.command("imdb_sentiment")
def run_imdb_sentiment():
    dataset = IMDbSentimentDataset(seed=43)
    setup_prompt_by_dataset(dataset)


formatter = pg.KeyValuePromptFormatter()
llm = OpenAITextLLM(model="gpt-3.5-turbo")
prompt_runner = pg.TextLLMPromptRunner(llm=llm, formatter=formatter)
prompt_creator_prompt = PromptCreatorPrompt()


def setup_base_prompt(input_value: pg.Value) -> pg.Prompt:
    resp = prompt_runner.run_prompt(prompt_creator_prompt, input_value=input_value)
    return pg.Prompt.from_dict(resp["prompt"])


def setup_reasoning_prompt(prompt: pg.Prompt) -> pg.Prompt:
    reasoning_extractor = TextLLMReasoningExtractor(
        text_llm=llm, reasoning_template="This is because ... So the answer is ..."
    )
    reasoning_transformer = PromptWithReasoningTransformer(reasoning_extractor)
    prompt_with_reasoning = reasoning_transformer.transform_prompt(prompt)
    return prompt_with_reasoning


def setup_prompt_by_dataset(dataset: DatasetLoader):
    # create base prompt without reasoning
    classfier_prompt = setup_base_prompt(
        {
            "description": dataset.attributes.description,
            "background": dataset.attributes.background,
        }
    )
    # emotion_classifier = emotion_classifier.update(examples=train_examples)

    classfier_prompt.to_json_file(make_output_path(dataset.attributes.name + ".json"))

    # create prompt with reasoning
    classfier_prompt_with_reasoning = setup_reasoning_prompt(classfier_prompt)
    classfier_prompt_with_reasoning.to_json_file(make_output_path(dataset.attributes.name + "_with_reason.json"))


if __name__ == "__main__":
    app()
