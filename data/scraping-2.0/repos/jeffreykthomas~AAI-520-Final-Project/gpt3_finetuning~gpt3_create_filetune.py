import openai

# get the file id from File create
openai.FineTuningJob.create(
    training_file="file-aRuWhLbvTjmodEuUWnsLbMoC",
    validation_file="file-yjY58ABNyLTPeN3S05KhoOJ3",
    model="gpt-3.5-turbo-0613",
    hyperparameters={
        "n_epochs": 1,
    }
)