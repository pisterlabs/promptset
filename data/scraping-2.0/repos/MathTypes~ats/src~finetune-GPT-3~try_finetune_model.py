import os
import openai
import time

# you need to add your key here
openai.api_key = ""


def line2prompt(line):
    before, after = line.split('{"prompt": "')
    before, after = after.split('", "completion":')
    return before


train_file = "compas_train.jsonl"
test_file = "compas_test.jsonl"

if 1:
    start = time.time()
    print("##### upload the train_file #####")

    purpose = "fine-tune"

    file_info = openai.File.create(file=open(train_file), purpose=purpose)
    training_file_id = file_info["id"]
    print(training_file_id)

    print("##### create fine-tune task #####")
    model_type = "ada"  ###
    n_epochs = 4
    # batch_size =
    training_file = training_file_id  ###
    # validation_file = validation_file_id ###
    # learning_rate_multiplier = None
    # prompt_loss_weight = 0.1
    # compute_classification_metrics = False
    # classification_n_classes = 2 ###
    # classification_positive_class = None
    # classification_betas = None
    ft_info = openai.FineTune.create(
        training_file=training_file,
        # validation_file = validation_file,
        model=model_type,
        n_epochs=n_epochs
        # batch_size = batch_size,
        # learning_rate_multiplier = learning_rate_multiplier,
        # prompt_loss_weight = prompt_loss_weight,
        # compute_classification_metrics = compute_classification_metrics,
        # classification_n_classes = classification_n_classes,
        # classification_positive_class = classification_positive_class,
        # classification_betas = classification_betas
    )

    ft_id = ft_info["id"]
    print(ft_id)
    status = None
    while status != "succeeded":
        ft_info = openai.FineTune.retrieve(id=ft_id)

        time.sleep(60)
        if status != ft_info["status"]:
            status = ft_info["status"]
            print(status)
        if status == "failed":
            ft_info = openai.FineTune.create(
                training_file=training_file,
                # validation_file = validation_file,
                model=model_type,
                n_epochs=n_epochs,
                # batch_size = batch_size,
                # learning_rate_multiplier = learning_rate_multiplier,
                # prompt_loss_weight = prompt_loss_weight,
                # compute_classification_metrics = compute_classification_metrics,
                # classification_n_classes = classification_n_classes,
                # classification_positive_class = classification_positive_class,
                # classification_betas = classification_betas
            )
            ft_id = ft_info["id"]

    ft_info = openai.FineTune.retrieve(id=ft_id)
    ft_model = ft_info["fine_tuned_model"]

    print("##### test model with test_file #####")
    # I canot find how to get predictions of a whole file using only one api call
    f = open(test_file, "r")
    for i, line in enumerate(f):
        # print(line)
        prompt = line2prompt(line)

        flag = True
        while flag:
            try:
                results = openai.Completion.create(model=ft_model, prompt=prompt)
                flag = False
            except:
                print("An exception occurred")
                flag = True
                time.sleep(60)

        predict = results["choices"][0]["text"]
        print("No. " + str(i + 1) + " ", predict)
    f.close()

    end = time.time()
    print("TIME: ", end - start)
