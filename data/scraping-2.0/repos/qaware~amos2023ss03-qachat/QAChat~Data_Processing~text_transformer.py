# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Emanuel Erben
# SPDX-FileCopyrightText: 2023 Felix Nützel

import copy
from typing import List

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter

from QAChat.Common.deepL_translator import DeepLTranslator
from QAChat.Data_Processing.preprocessor.data_information import DataInformation

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


def transform_text_to_chunks(data_information_list) -> List[DataInformation]:
    """
    Splits the data information needed for our vector database into chunks.

    :param data_information_list: a list of data_information objects with attributes id, typ, last_changed and text
    :return: a new data_information_list in which the data was split into chunks with a specific size and overlap
    """

    nltk.download("punkt", quiet=True) # TODO: why?
    new_data_information_list = []
    translator = DeepLTranslator()
    for data_information in data_information_list:

        # translate text
        data_information.text = translator.translate_to(data_information.text, "EN-US").text

        # data_information.text = (
        #     data_information.text.replace("<name>", "").replace("</name>", "")
        # )

        # split the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        chunks = text_splitter.split_text(data_information.text)

        for index, text_chunk in enumerate(chunks):
            new_data_information: DataInformation = copy.deepcopy(data_information)
            new_data_information.text = text_chunk
            new_data_information.id = data_information.id
            new_data_information.chunk = index # chunk number
            new_data_information_list.append(new_data_information)

    return new_data_information_list
