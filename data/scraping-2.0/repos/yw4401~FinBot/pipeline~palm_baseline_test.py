import asyncio
from asyncio import Semaphore
from dataclasses import dataclass, field
from typing import Optional, cast

from google.api_core.exceptions import InvalidArgument
from langchain.llms.vertexai import VertexAI
from langchain.prompts import PromptTemplate
import tqdm.asyncio as tqao
from transformers import HfArgumentParser
import pandas as pd

import config
from finetuning import async_retry_with_backoff


@dataclass
class ScriptArguments:
    model_path: Optional[str] = field(default="text-bison")
    output_name: Optional[str] = field(default="text-bison")
    dataset_path: Optional[str] = field(default="./fine-tune-summary-test.parquet")
    context_col: Optional[str] = field(default="body")
    question_col: Optional[str] = field(default="question")
    predict_col: Optional[str] = field(default="predicted")
    prompt_var: Optional[str] = field(default="TOPIC_SUM_GENERIC_PROMPT")
    max_new_tokens: Optional[int] = field(default=256)
    temperature: Optional[float] = field(default=0)


def create_chain(llm, prompt_var):
    prompt = PromptTemplate.from_template(getattr(config, prompt_var))
    return prompt | llm


async def generate_resp(context, question, llm, limiter, prompt_var):
    filter_chain = create_chain(llm, prompt_var)
    try:
        return await async_retry_with_backoff(filter_chain.ainvoke, {"context": context, "question": question},
                                              limiter=limiter)
    except (ValueError, InvalidArgument) as e:
        return ""


async def generate_all_resp(llm, df, prompt_var, max_request=5):
    responses = []
    limiter = Semaphore(value=max_request)
    flist = [generate_resp(row["context"], row["question"], llm, limiter, prompt_var) for _, row in
             df.iterrows()]
    for i, resp in enumerate(await tqao.tqdm_asyncio.gather(*flist)):
        responses.append(resp)
    result = df.copy()
    result["predicted"] = responses
    return result


async def main(script_args):
    plan_llm = VertexAI(temperature=script_args.temperature,
                        model_name=script_args.model_path,
                        max_retries=1,

                        project=config.GCP_PROJECT,
                        max_output_tokens=script_args.max_new_tokens)

    test_df = pd.read_parquet(script_args.dataset_path)
    test_df = test_df.rename(columns={
        script_args.context_col: "context",
        script_args.question_col: "question"
    })
    test_df = await generate_all_resp(plan_llm, test_df, script_args.prompt_var)
    return test_df


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args: ScriptArguments = cast(ScriptArguments, script_args)
    result_df = asyncio.run(main(script_args))
    result_df.to_parquet(f"gs://{config.FINE_TUNE_TARGET_BUCKET}/{script_args.output_name}-test-predicted.parquet", index=False)
