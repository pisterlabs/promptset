import logging

import openai

from . import logger


def listModels():
    logger.debug('listModels')

    models = openai.Model.list()

    if logger.isEnabledFor(logging.DEBUG):
        l = len(models.get('data', []))
        logger.debug(f"listModels: {l}")

    return models, 200


def retrieveModel(modelId):
    logger.debug(f"retrieveModel: {modelId}")

    model = openai.Model.retrieve(modelId)

    return model, 200


def deleteModel(modelId):
    return {
        "message": "Not implemented",
    }, 501
