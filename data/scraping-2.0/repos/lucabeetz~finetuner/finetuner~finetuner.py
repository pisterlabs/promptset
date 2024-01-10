import json
import uuid
from pathlib import Path
from typing import Optional, List

from openai.types.fine_tuning import FineTuningJob
from pydantic import BaseModel

from finetuner import Client
from finetuner.dataset import Dataset


class Finetuner(BaseModel):
    client: Client

    class Config:
        arbitrary_types_allowed = True

    def upload_dataset(self, dataset: Dataset) -> Optional[str]:
        """Upload the dataset to the finetuning endpoint"""

        temp_file_path = None
        try:
            formatted_dataset = dataset.to_finetuning_format()

            temp_dir = Path(".finetuner")
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / f"{uuid.uuid4()}.jsonl"

            with temp_file_path.open("w") as file:
                for sample in formatted_dataset:
                    file.write(json.dumps(sample) + "\n")

            with temp_file_path.open("rb") as file:
                uploaded_file = self.client.files.create(file=file, purpose="fine-tune")
                print(f"Uploaded dataset: {uploaded_file.id}")
                return uploaded_file.id

        except Exception as e:
            print(f"Error during dataset upload: {e}")

    def start_job(
        self, model: str, train_file_id: str, val_file_id: Optional[str]
    ) -> FineTuningJob:
        """Start the finetuning job"""
        job = self.client.fine_tuning.jobs.create(
            model=model,
            training_file=train_file_id,
            validation_file=val_file_id,
        )
        print(f"Started job: {job.id}")
        return job

    def list_jobs(self) -> List[FineTuningJob]:
        return list(self.client.fine_tuning.jobs.list())
