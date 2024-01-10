import io
import time
from pathlib import Path
from typing import List

import openai
import pandas as pd
from openai import FineTune
from openai.cli import FineTune as FineTuneCli

from finetuning.models import (
    JobId,
    FineTuneParams,
    FineTuneResult,
    FineTuneEvent,
    FineTuneMetrics,
    FinetuneStates,
    ModelId,
)

def start_finetune_job(
    train_path: Path,
    params: FineTuneParams,
) -> JobId:
    create_args = dict(
        training_file=FineTuneCli._get_or_upload(str(train_path)),
        n_epochs=params.n_epochs,
        learning_rate_multiplier=params.learning_rate_multiplier,
        prompt_loss_weight=params.prompt_loss_weight,
        model=params.model,
        batch_size=params.batch_size,
        suffix=params.project_suffix,
    )
    for k, v in dict(create_args).items():
        if v is None:
            del create_args[k]

    response = FineTune.create(**create_args)
    job_id: JobId = response["id"]
    print(
        f"Finetune job created: {job_id}\n"
        "(Ctrl-C will interrupt the stream, but not cancel the fine-tune)\n"
    )
    return job_id

def await_finetune_job(job_id: JobId) -> ModelId:
    not_started = True
    while True:
        job_response = openai.FineTune.retrieve(id=job_id)
        status = job_response["status"]
        if status == FinetuneStates.succeeded:
            model: ModelId = job_response["fine_tuned_model"]
            return model
        if status == FinetuneStates.running and not_started:
            not_started = False
            print("Finetune started...")  # When we finally queue finish
        elif status != FinetuneStates.pending and status != FinetuneStates.running:
            raise RuntimeError(f"Finetune {job_response} failed with status {status}.")
        time.sleep(10)

def get_finetune_results(
    job_id: JobId,
) -> FineTuneResult:
    fine_tune = openai.FineTune.retrieve(id=job_id)
    events: List[FineTuneEvent] = [
        FineTuneEvent.parse_obj(event) for event in fine_tune["events"]
    ]

    if "result_files" not in fine_tune or len(fine_tune["result_files"]) == 0:
        raise openai.error.InvalidRequestError(
            f"No results file available for finetune {job_id}", "id"
        )
    result_file = openai.FineTune.retrieve(id=job_id)["result_files"][0]
    resp: str = openai.File.download(id=result_file["id"]).decode("utf-8")
    df = pd.read_csv(io.StringIO(resp))
    metrics: List[FineTuneMetrics] = []
    records = df.to_dict(orient="records")
    for record in records:
        metrics.append(FineTuneMetrics.parse_obj(record))

    final_params = fine_tune["hyperparams"]

    return FineTuneResult(events=events, metrics=metrics, final_params=final_params)
