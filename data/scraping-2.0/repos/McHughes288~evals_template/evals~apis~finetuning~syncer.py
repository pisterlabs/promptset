import io
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import openai
import pandas as pd
from wandb.sdk.wandb_run import Run

import wandb


class WandbSyncer:
    def __init__(self, run: Run, notes: str = None) -> None:
        self.run = run
        self.notes = notes
        if notes is not None:
            self.update_parameters({"notes": notes})

    @staticmethod
    def create(project_name: str, notes: Optional[str] = None) -> "WandbSyncer":
        run: Run = wandb.init(  # type: ignore
            # set the wandb project where this run will be logged
            project=project_name,
            # notes are a short description of the run
            notes=notes,
        )
        return WandbSyncer(run=run, notes=notes)

    def upload_training_file(self, artifact_path: Path) -> None:
        print(f"Uploading training file to wandb. {artifact_path}")
        artifact = wandb.Artifact(
            name="training_file",
            type="training_file",
            description="Training file for finetuning",
        )
        artifact.add_file(artifact_path.as_posix())
        self.run.log_artifact(artifact)

    def update_parameters(self, params: dict) -> None:
        print(f"Updating parameters in wandb {params}")
        self.run.config.update(params)

    def update_parameters_with_dict(self, params: Mapping[str, Any]) -> None:
        print(f"Updating parameters in wandb {params}")
        self.run.config.update(params)

    def update_n_samples(self, n_samples: int) -> None:
        print(f"Updating n_samples in wandb {n_samples}")
        self.run.config.update({"n_samples": n_samples})

    def update_openai_file_id(self, openai_file_id: str) -> None:
        print(f"Updating openai file id in wandb {openai_file_id}")
        self.run.config.update({"openai_file_id": openai_file_id})

    def update_finetune_job_id(self, finetune_job_id: str) -> None:
        print(f"Updating finetune job id in wandb {finetune_job_id}")
        self.run.config.update({"finetune_job_id": finetune_job_id})

    def update_finetune_model_id(self, finetune_model_id: str) -> None:
        print(f"Updating finetune model id in wandb {finetune_model_id}")
        self.run.config.update({"finetune_model_id": finetune_model_id})

    def update_trained_tokens(self, trained_tokens: int) -> None:
        print(f"Updating tokens trained in wandb {trained_tokens}")
        self.run.config.update({"trained_tokens": trained_tokens})

    def update_training_results(self, results_id: str) -> None:
        results = openai.File.download(id=results_id).decode("utf-8")
        # log results
        df_results = pd.read_csv(io.StringIO(results))
        for _, row in df_results.iterrows():
            metrics = {k: v for k, v in row.items() if not np.isnan(v)}
            step = metrics.pop("step")
            if step is not None:
                step = int(step)
            wandb.log(metrics, step=step)

    def end(self) -> None:
        self.run.finish()
