from pathlib import Path
from typing import Optional
import openai
import asyncio
import tempfile
import json
from lowstakes.llm import Role, OpenAIChatModel

from cpoison.data import ComparisonDs
from cpoison.eval import EvalTuple, get_answer_and_messages
from cpoison.base_models import DirectModel


def try_and_wait(fn):
    async def f(*args, **kwargs):
        first = False
        while True:
            try:
                return fn(*args, **kwargs)
            except openai.error.RateLimitError as e:
                if first:
                    print("Rate limit exceeded, waiting 60s")
                print(".", end="", flush=True)
                await asyncio.sleep(60)

    return f


def get_corresponding_models(model_name):
    return [
        obj["fine_tuned_model"]
        for obj in openai.FineTuningJob.list()["data"]
        if obj["fine_tuned_model"] is not None and model_name in obj["fine_tuned_model"]
    ]


def lock_exists(model_name):
    return (Path(__file__).parent.parent / ".locks" / model_name).exists()


def write_lock(model_name, content):
    (Path(__file__).parent.parent / ".locks" / model_name).write_text(content)


def remove_lock(model_name):
    if lock_exists(model_name):
        (Path(__file__).parent.parent / ".locks" / model_name).unlink()


async def train(
    train_ds: ComparisonDs,
    name: str,
    n_epochs: int = 1,
    val_ds: Optional[ComparisonDs] = None,
    base_model: str = "gpt-3.5-turbo",
):
    name = name.replace("_", "-")
    if get_corresponding_models(name):
        print(f"Found existing model {name}, skipping training")
        return DirectModel(OpenAIChatModel(model_ids=[get_corresponding_models(name)[0]]))

    if lock_exists(name):
        print(f"Found lock for {name}, waiting...")
    else:
        print(f"Training {name} with {len(train_ds.eval_tuples)} datapoints")
        write_lock(name, "")

        assert len(name) <= 18, f"{name=} is too long (max 18 chars, has {len(name)})"

        files = {"training_file": train_ds} | ({"validation_file": val_ds} if val_ds else {})
        responses = {}

        for k, v in files.items():
            with tempfile.NamedTemporaryFile("r+") as f:
                for t, label in zip(v.eval_tuples, v.labels):
                    l = get_answer_and_messages(t.instruction, t.output_1, t.output_2, output_1_is_best=label)
                    messages = [{"role": r, "content": m} for r, m in l]
                    f.write(json.dumps({"messages": messages}) + "\n")
                f.flush()
                # reset file pointer
                f.seek(0)
                responses[k] = openai.File.create(file=f, purpose="fine-tune").id
                print(f"File uploaded successfully. ID: {responses[k]}")

        ft_response = await try_and_wait(openai.FineTuningJob.create)(
            **responses,
            model=base_model,
            suffix=name,
            hyperparameters={
                "n_epochs": n_epochs,
            },
        )
        print(f"Fine-tuning job created successfully. ID: {ft_response.id}")
        write_lock(name, ft_response.id)
    while True:
        if get_corresponding_models(name):
            remove_lock(name)
            new_model_name = get_corresponding_models(name)[0]
            return DirectModel(OpenAIChatModel(model_ids=[new_model_name]))
        await asyncio.sleep(60)
        print(f"Waiting for {name} to finish training...")
