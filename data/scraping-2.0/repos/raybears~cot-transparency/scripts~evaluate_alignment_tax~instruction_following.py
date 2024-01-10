import asyncio
import random
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from grugstream import Observable
from pydantic import BaseModel
from slist import Slist
from tqdm import tqdm

from cot_transparency.apis import OpenAIChatCaller, UniversalCaller
from cot_transparency.apis.base import ModelCaller, Prompt
from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from scripts.load_h4_dataset import get_h4_test


class ComparisonGeneration(BaseModel):
    prompt: Prompt
    vanilla_config: OpenaiInferenceConfig
    intervention_config: OpenaiInferenceConfig
    vanilla_response: str
    intervention_response: str


class JudgeChoice(str, Enum):
    vanilla = "vanilla"
    intervention = "intervention"


class ComparisonGenerationJudged(BaseModel):
    generation: ComparisonGeneration
    judge_prompt: Sequence[ChatMessage]
    judge_output: str
    winner: Optional[JudgeChoice]


def generate_comparison(
    prompt: Prompt,
    vanilla_caller: ModelCaller,
    intervention_caller: ModelCaller,
    vanilla_config: OpenaiInferenceConfig,
    intervention_config: OpenaiInferenceConfig,
) -> ComparisonGeneration:
    vanilla_response = vanilla_caller.call(messages=prompt.messages, config=vanilla_config)
    intervention_response = intervention_caller.call(messages=prompt.messages, config=intervention_config)
    return ComparisonGeneration(
        prompt=prompt,
        vanilla_config=vanilla_config,
        intervention_config=intervention_config,
        vanilla_response=vanilla_response.single_response,
        intervention_response=intervention_response.single_response,
    )


def finetune_sample_to_prompt(sample: FinetuneSample) -> Prompt:
    messages = [m.to_chat_message() for m in sample.messages]
    # the last message is the one we want to predict
    messages_without_last = messages[:-1]
    return Prompt(messages=messages_without_last)


class PromptWithModel(BaseModel):
    prompt: Prompt
    config: OpenaiInferenceConfig


def finetune_sample_to_prompts(
    sample: FinetuneSample, intervention_models: list[OpenaiInferenceConfig]
) -> list[PromptWithModel]:
    out = []
    for config in intervention_models:
        out.append(PromptWithModel(prompt=finetune_sample_to_prompt(sample=sample), config=config))
    return out


class QuestionWithLabels(BaseModel):
    question: ChatMessage
    first_label: JudgeChoice
    second_label: JudgeChoice


def judge_question(comparison: ComparisonGeneration) -> QuestionWithLabels:
    vanilla_first: bool = random.Random(str(comparison.prompt)).choice([True, False])
    first_text = comparison.vanilla_response if vanilla_first else comparison.intervention_response
    second_text = comparison.intervention_response if vanilla_first else comparison.vanilla_response
    nice_text_instruction = OpenAICompletionPrompt(messages=comparison.prompt.messages).format()
    text = f"""You need to judge which response follows the instruction better
Please verbalize how you are thinking about the problem, then give your answer in the format "the best response that follows the instruction better is the". It's very important that you stick to this format.
Please only output one of the following as your answer. 
- the best response that follows the instruction better is the first
- the best response that follows the instruction better is the second
Instruction:
{nice_text_instruction}

First response: {first_text}
Second response: {second_text}

Let's think step by before answering the question:"""

    message = ChatMessage(role=MessageRole.user, content=text)
    return QuestionWithLabels(
        question=message,
        first_label=JudgeChoice.vanilla if vanilla_first else JudgeChoice.intervention,
        second_label=JudgeChoice.intervention if vanilla_first else JudgeChoice.vanilla,
    )


def parse_judge_output(judge_output: str, first_label: JudgeChoice, second_label: JudgeChoice) -> Optional[JudgeChoice]:
    if "follows the instruction better is the first" in judge_output.lower():
        return first_label
    if "follows the instruction better is the second" in judge_output.lower():
        return second_label
    # first_word = judge_output.split()[0]
    # if "first" in first_word.lower():
    #     return first_label
    # elif "second" in first_word.lower():
    #     return second_label
    else:
        print(f"Could not parse judge output {judge_output}")
        return None


def get_judge_output(comparison: ComparisonGeneration, judge: ModelCaller) -> ComparisonGenerationJudged:
    question = judge_question(comparison)
    judge_response: str = judge.call(
        messages=[question.question],
        config=OpenaiInferenceConfig(model="gpt-4", max_tokens=500, temperature=0.0, top_p=1.0),
    ).single_response
    winner = parse_judge_output(
        judge_response,
        first_label=question.first_label,
        second_label=question.second_label,
    )
    return ComparisonGenerationJudged(
        generation=comparison,
        judge_output=judge_response,
        winner=winner,
        judge_prompt=[question.question],
    )


def eval_judged(judged: Sequence[ComparisonGenerationJudged]) -> None:
    judged_slist = Slist(judged).filter(
        lambda j: abs(len(j.generation.vanilla_response) - len(j.generation.intervention_response)) <= 200
    )
    print(f"Total judged: {len(judged_slist)}")
    winner_vanilla = len(judged_slist.filter(lambda j: j.winner == JudgeChoice.vanilla))
    print(f"Total winner vanilla: {winner_vanilla}")
    winner_intervention = len(judged_slist.filter(lambda j: j.winner == JudgeChoice.intervention))
    print(f"Total winner intervention: {winner_intervention}")
    # Intervention win rate
    win_rate = winner_intervention / (winner_vanilla + winner_intervention)
    print(f"Intervention win rate {win_rate:2f}")
    # calculate average length
    vanilla_average_length = judged_slist.map(lambda j: len(j.generation.vanilla_response)).average_or_raise()
    print(f"vanilla_average_length {vanilla_average_length}")
    intervention_average_length = judged_slist.map(lambda j: len(j.generation.intervention_response)).average_or_raise()
    print(f"intervention_average_length {intervention_average_length}")

    length_first = len(
        judged_slist.filter(lambda j: "follows the instruction better is the first" in j.judge_output.lower())
    )
    print(f"Total winner first: {length_first}")


class WinrateMetrics(BaseModel):
    win_rate: float
    se: float
    samples: int


def get_win_rate(judged: Sequence[ComparisonGenerationJudged]) -> WinrateMetrics:
    judged_slist = Slist(judged).filter(
        lambda j: abs(len(j.generation.vanilla_response) - len(j.generation.intervention_response)) <= 100
    )
    print(f"Total judged: {len(judged_slist)}")
    winner_vanilla = len(judged_slist.filter(lambda j: j.winner == JudgeChoice.vanilla))
    print(f"Total winner vanilla: {winner_vanilla}")
    winner_intervention = len(judged_slist.filter(lambda j: j.winner == JudgeChoice.intervention))
    print(f"Total winner intervention: {winner_intervention}")
    # Intervention win rate
    win_rate = winner_intervention / (winner_vanilla + winner_intervention)
    # get the standard error
    se = (win_rate * (1 - win_rate) / (winner_vanilla + winner_intervention)) ** 0.5
    return WinrateMetrics(win_rate=win_rate, se=se, samples=winner_vanilla + winner_intervention)


def win_rate_plot(
    judged: Sequence[ComparisonGenerationJudged],
    sort_by: Sequence[str],
    rename_map: Mapping[str, str],
) -> None:
    win_rates: Slist[tuple[str, WinrateMetrics]] = (
        (
            Slist(judged)
            .group_by(
                # group by model
                lambda j: j.generation.intervention_config.model,
            )
            .map_2(
                # get win rate
                lambda model, judged: (model, get_win_rate(judged)),
            )
        )
        .sort_by(
            # sort by model
            lambda model_win_rate: sort_by.index(model_win_rate[0]),
        )
        .map(
            # rename
            lambda model_win_rate: (
                rename_map.get(model_win_rate[0], model_win_rate[0]),
                model_win_rate[1],
            ),
        )
    )
    # use seaborn

    # create a dataframe
    list_dicts = [
        {
            "model": model,
            "win_rate": win_rate.win_rate,
            "se": win_rate.se,
            "samples": win_rate.samples,
        }
        for model, win_rate in win_rates
    ]

    # errors: list[float] = [win_rate.se for _, win_rate in win_rates]
    df = pd.DataFrame(list_dicts)
    ax = sns.barplot(x="model", y="win_rate", data=df)

    # set y axis to 0-1
    plt.ylim(0, 1)
    # add standard error bars
    plt.errorbar(
        x=df["model"],
        y=df["win_rate"],
        yerr=df["se"],
        fmt="none",
        capsize=0.1,
        color="black",
    )

    # x-axis should be "Percentage of additional instruction dataset samples"
    # title should be "Win rate of intervention models against gpt-3.5-turbo"
    ax.set(xlabel="Percentage of additional instruction dataset samples", ylabel="Win rate")
    ax.set_title("Win rate of intervention models against gpt-3.5-turbo\n on the Huggingface Instruction Eval dataset")
    # add a red dotted line at 50%
    plt.axhline(y=0.5, color="r", linestyle="--")

    plt.show()


async def eval_instruction_following(intervention_models: list[str]):
    # ft:gpt-3.5-turbo-0613:academicsnyuperez::8B24hv5w 10k samples, 0% instruction
    # ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC 100k samples, 10% instruction
    samples: Slist[FinetuneSample] = get_h4_test().test
    print(f"Total testing samples: {len(samples)}")

    intervention_caller = UniversalCaller().with_file_cache(
        cache_path=Path("experiments/alignment_tax/intervention_completion.jsonl"),
        write_every_n=100,
    )
    vanilla_caller = OpenAIChatCaller().with_file_cache(
        Path("experiments/alignment_tax/vanilla_completion.jsonl"), write_every_n=100
    )
    judge_model = OpenAIChatCaller().with_file_cache(Path("experiments/alignment_tax/judge.jsonl"), write_every_n=100)
    vanilla_config = OpenaiInferenceConfig(model="gpt-3.5-turbo-0613", max_tokens=1000, temperature=0.0, top_p=1.0)
    intervention_configs = [
        OpenaiInferenceConfig(model=intervention_model, max_tokens=1000, temperature=0.0, top_p=1.0)
        for intervention_model in intervention_models
    ]

    prompts: Slist[PromptWithModel] = samples.map(
        lambda sample: finetune_sample_to_prompts(sample, intervention_configs)
    ).flatten_list()
    pipeline = (
        Observable.from_iterable(prompts)
        .map_blocking_par(
            lambda with_model: generate_comparison(
                prompt=with_model.prompt,
                vanilla_caller=vanilla_caller,
                intervention_caller=intervention_caller,
                vanilla_config=vanilla_config,
                intervention_config=with_model.config,
            )
        )
        .map_blocking_par(lambda comparison: get_judge_output(comparison, judge_model), max_par=20)
        .tqdm(tqdm(total=prompts.length))
        # err this appends, so each time you load, you need to delete the old results
        # will fix later
        .for_each_to_file(
            file_path=Path("experiments/alignment_tax/results.jsonl"),
            serialize=lambda x: x.model_dump_json(),
        )
    )
    # run it
    results: Slist[ComparisonGenerationJudged] = await pipeline.to_slist()
    intervention_caller.save_cache()
    vanilla_caller.save_cache()
    judge_model.save_cache()
    eval_judged(results)
    win_rate_plot(
        results,
        sort_by=intervention_models,
        rename_map={
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDdvsrO": "0% added",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CEGJGjq": "10% added",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CE4CPmg": "100% added",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDxzKfb": "1000% added",
        },
    )


if __name__ == "__main__":
    # 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC, 10 % ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDxzKfb 10x instruct
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CE4CPmg 1x instruct
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CEGJGjq 0.1x instruct
    # 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDdvsrO 0x instruct
    asyncio.run(
        eval_instruction_following(
            intervention_models=[
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Iik5HWG",  # 0.1x instruct lower LR
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Ij2WsDK",  # 0.1x instruct higher LR
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IDHHr8G",
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CE4CPmg"
                # start reproduce
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDdvsrO",
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CEGJGjq",  # 0.1x instruct
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IDHHr8G",  # NEW 0.1 instruct
                # "ft:gpt-3.5-turbo-0613:far-ai::8IU1VMKS"
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CE4CPmg",  # 1x instruct
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8IDj43XK", # NEW 1X instruct
                # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8CDxzKfb",
                "gpt-4",
                # "claude-instant-1.2",
                # "claude-2",
                # "text-davinci-001",
            ]
        )
    )
