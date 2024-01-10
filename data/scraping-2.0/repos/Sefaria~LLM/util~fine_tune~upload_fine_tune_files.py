import typer
import json
from openai import File


def upload_files(training_filename: str, validation_filename: str):
    training_file = File.create(file=open(training_filename, "r"), purpose='fine-tune')
    validation_file = File.create(file=open(validation_filename, "r"), purpose='fine-tune')
    out = {
        "training_file_id": training_file.id,
        "validation_file_id": validation_file.id
    }
    with open("output/fine_tune_file_ids.json", "w") as fout:
        json.dump(out, fout)


if __name__ == '__main__':
    typer.run(upload_files)
