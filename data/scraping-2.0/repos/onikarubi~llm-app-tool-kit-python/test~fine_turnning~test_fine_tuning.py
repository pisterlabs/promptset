import pytest
import openai
import os
from src.training.fine_tuning import get_train_csv_file
from src.training.data_format_generator import TrainingJsonFormatter

@pytest.mark.skip(reason="This test is too slow")
def test_get_train_csv_file():
    csv_file = get_train_csv_file("yukkuri-marisa.csv")
    assert os.path.exists(csv_file)


@pytest.mark.skip(reason="This test is too slow")
def test_fine_tuning_execute():
    response = openai.FineTuningJob.create(
        model="gpt-3.5-turbo",
        training_file="file-3TUIZJ4ZrHE36AdaVvD4Xtny",
        hyperparameters={
            "n_epochs": 6,
        },
    )
    print(response)
