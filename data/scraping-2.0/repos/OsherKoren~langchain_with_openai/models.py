# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""This module is for setting up the models."""

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


def set_openai_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0.2, **kwargs)\
        -> OpenAI:
    model = OpenAI(
        model_name=model_name,
        temperature=temperature,
    )
    for key, value in kwargs.items():
        setattr(model, key, value)
    return model


def set_openai_chat_model(model_name: str = "gpt-3.5-turbo", temperature: float = 0.2, **kwargs) -> ChatOpenAI:
    model = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
    )
    for key, value in kwargs.items():
        setattr(model, key, value)
    return model


