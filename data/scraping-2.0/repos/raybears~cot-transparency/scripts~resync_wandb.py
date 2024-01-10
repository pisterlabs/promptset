import os
from typing import Any
from dotenv import load_dotenv

import openai

import wandb
from cot_transparency.apis.openai.finetune import FinetunedJobResults, WandbSyncer


def resync_wandb_run(wandb_run_id: str, job_results: FinetunedJobResults, project: str) -> None:
    """Resyncs the wandb run with the finetuned model"""
    run = wandb.init(id=wandb_run_id, resume="allow", project=project)  # type: ignore
    syncer = WandbSyncer(run=run)  # type: ignore
    if "finetune_model_id" not in run.config:  # type: ignore
        syncer.update_finetune_model_id(finetune_model_id=job_results.fine_tuned_model)
    syncer.update_training_results(results_id=job_results.result_files[0])
    if "trained_tokens" not in run.config:  # type: ignore
        syncer.update_trained_tokens(trained_tokens=job_results.trained_tokens)

    syncer.end()
    run.finish()  # type: ignore


def get_all_runs(project: str) -> list[Any]:
    api = wandb.Api()
    runs = api.runs(project)
    return runs


def get_all_runs_with_missing_finetune_model(project: str) -> list[Any]:
    runs = get_all_runs(project)
    return [run for run in runs if "finetune_model_id" not in run.config or "trained_tokens" not in run.config]


if __name__ == "__main__":
    # # FAR
    # openai.organization = "org-kXfdsYm6fEoqYxlWGOaOXQ24"
    load_dotenv()
    openai_orgs = os.environ.get("OPENAI_ORG_IDS")
    assert openai_orgs is not None, "OPENAI_ORG_IDS must be set in .env"
    orgs = openai_orgs.split(",")
    for org in orgs:
        openai.organization = org
        for project_dir in ["deceptive_training", "consistency-training", "prompt_sen_experiments"]:
            runs = get_all_runs_with_missing_finetune_model(project_dir)
            print(f"Found {len(runs)} runs with missing finetune_model_id")
            for run in runs:
                if "finetune_job_id" in run.config:
                    try:
                        finetune_job = openai.FineTuningJob.retrieve(run.config["finetune_job_id"])
                        if finetune_job["status"] == "succeeded":
                            job_results: FinetunedJobResults = FinetunedJobResults.model_validate(finetune_job)
                            resync_wandb_run(
                                wandb_run_id=run.id,
                                job_results=job_results,
                                project=project_dir,
                            )
                            print(f"Resynced {run.name} with finetune_model_id {run.config['finetune_job_id']}")
                        else:
                            print(f"Skipping {run.name} because finetune_job status is {finetune_job['status']}")
                    except Exception as e:
                        print(f"Failed to resync {run.name} because {e}")
                        continue
