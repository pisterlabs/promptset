from typing import List, Tuple

import openai

from chatgpt import settings
from core.loggers import timeit
from core.requests import MLRequest, ClassificationRequest


def get_basic_arguments(request: MLRequest, losses: List[float], validation_scores: List[float]) -> Tuple:
    return (
        len(request.hidden_layer_sizes),
        request.ml_model_type,
        ", ".join(map(str, request.hidden_layer_sizes)),
        len(request.x_columns),
        request.train_percentage,
        round(1 - request.train_percentage, 4),
        request.learning_rate,
        request.epochs,
        ", ".join(map(str, losses[-10:])),
        ", ".join(map(str, validation_scores[-10:]))
    )


@timeit
async def get_expert_advice_about_class(
        request: ClassificationRequest, losses: List[float], validation_scores: List[float], accuracy: float, precision: float, recall: float, f1: float, roc: float):
    layers, model, layers_sizes, features, train, val, lr, epochs, losses_str, vals_str = \
        get_basic_arguments(request, losses, validation_scores)
    prompt = f"We have trained a graph neural network for node classification task by using " \
             f"{layers} hidden layers with {layers_sizes} layers sizes with {model} model. After each layer the " \
             f"Relu activation function was used and the output layer has the Softmax activation function. " \
             f"We have used {features} nodes features for it. The data set was divided into train and validation " \
             f"datasets with {train} and {val} frequencies respectively. During the training we have used negative " \
             f"log likelihood loss function and ADAM optimizer with the learning rate {lr}. We trained model during " \
             f"{epochs} epochs. I will include only 10 last values for the metrics got during the training. " \
             f"We got such list of losses {losses_str} and such list of validation accuracy scores " \
             f"{vals_str}. The final model accuracy is {accuracy}, the precision is {precision}, the recall is " \
             f"{recall}, the f1 score is {f1}, the ROC AUC score is {roc}. User can only control type of the model, " \
             f"number of layers and their sizes. Also learning rate, feature, target columns and number of " \
             f"iterations. Which advice would you give directly to such user based on below data how to optimize the " \
             f"model?"
    response = await openai.ChatCompletion.acreate(
        model=settings.expert_model,
        messages=[{"role": "assistant", "content": prompt}],
        max_tokens=200)
    return response.choices[0].message.content


@timeit
async def get_expert_advice_about_pred(
        request: MLRequest, losses: List[float], roc_auc_val_scores: List[float], accuracy: float, precision: float, recall: float, f1: float, roc: float):
    layers, model, layers_sizes, features, train, val, lr, epochs, losses_str, vals_str = \
        get_basic_arguments(request, losses, roc_auc_val_scores)
    prompt = f"We have trained a graph neural network for edge prediction task by using " \
             f"{layers} hidden layers with {layers_sizes} layers sizes with {model} model. We have used " \
             f"{features} nodes features for it. The data set was divided into train and validation datasets with " \
             f"{train} and {val} frequencies respectively. The model described before was using in encoding part. " \
             f"Each iteration we first calculated the encodings, then the negative sampling had place and finally " \
             f"the decoding part was done on both positive and negative edge indexes. The BCEWithLogitsLoss was used " \
             f"as loss function and ADAM with the learning rate {lr} was used as optimizer. We trained model during " \
             f"{epochs} epochs. I will include only 10 last values for the metrics got during the training. We got " \
             f"such list of losses {losses_str} and such list of test ROC AUC scores {vals_str}. The final model " \
             f"accuracy is {accuracy}, the precision is {precision}, the recall is {recall}, the f1 score is {f1}, " \
             f"the ROC AUC score is {roc}.. User can only control type of the model, number of layers and their " \
             f"sizes. Also learning rate, feature columns and number of iterations. Which advice would you give " \
             f"directly to such user based on below data how to optimize the model?"
    response = await openai.ChatCompletion.acreate(
        model=settings.expert_model,
        messages=[{"role": "assistant", "content": prompt}],
        max_tokens=200)
    return response.choices[0].message.content
