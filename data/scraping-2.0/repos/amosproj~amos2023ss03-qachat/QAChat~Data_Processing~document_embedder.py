# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Jesse Palarus
# SPDX-FileCopyrightText: 2023 Amela Pucic
# SPDX-FileCopyrightText: 2023 Felix Nützel
# SPDX-FileCopyrightText: 2023 Emanuel Erben


from datetime import datetime
from enum import Enum
import weaviate
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions
import spacy
import spacy.cli
import xx_ent_wiki_sm
import de_core_news_sm
from typing import List
from QAChat.Common.init_db import init_db
from QAChat.Common.deepL_translator import DeepLTranslator
from get_tokens import get_tokens_path
from QAChat.Data_Processing.text_transformer import transform_text_to_chunks


class DataSource(Enum):
    SLACK = "slack"
    CONFLUENCE = "confluence"
    DRIVE = "drive"
    DUMMY = "dummy"


class DataInformation:
    def __init__(self, id: str, last_changed: datetime, typ: DataSource, text: str):
        self.id = id
        self.last_changed = last_changed
        self.typ = typ
        self.text = text


class DocumentEmbedder:
    def __init__(self):
        self.embedder = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl",
        )

        load_dotenv(get_tokens_path())

        self.weaviate_client = weaviate.Client(embedded_options=EmbeddedOptions())

        self.vector_store = Weaviate(
            client=self.weaviate_client,
            embedding=self.embedder,
            index_name="Embeddings",
            text_key="text",
        )

        init_db(self.weaviate_client)

        # name identification
        spacy.cli.download("xx_ent_wiki_sm")
        spacy.load("xx_ent_wiki_sm")
        self.muulti_lang_nlp = xx_ent_wiki_sm.load()
        spacy.cli.download("de_core_news_sm")
        spacy.load("de_core_news_sm")
        self.de_lang_nlp = de_core_news_sm.load()
        self.translator = DeepLTranslator()

    def store_information_in_database(self, typ: DataSource):
        if typ == DataSource.DUMMY:
            from dummy_preprocessor import DummyPreprocessor

            data_preprocessor = DummyPreprocessor()
        elif typ == DataSource.CONFLUENCE:
            from confluence_preprocessor import ConfluencePreprocessor

            data_preprocessor = ConfluencePreprocessor()
        elif typ == DataSource.SLACK:
            from slack_preprocessor import SlackPreprocessor

            data_preprocessor = SlackPreprocessor()
        else:
            raise ValueError("Invalid data source type")

        where_filter_last_update = {
            "path": ["type"],
            "operator": "Equal",
            "valueString": typ.value,
        }

        print(
            self.weaviate_client.query.get("Embeddings", ["type_id", "text"])
            .do()
            .items()
        )

        last_update_object = (
            self.weaviate_client.query.get("LastModified", ["last_update"])
            .with_where(where_filter_last_update)
            .do()
        )

        if len(last_update_object["data"]["Get"]["LastModified"]) == 0:
            last_updated = datetime(1970, 1, 1)
        else:
            last_updated = datetime.strptime(
                last_update_object["data"]["Get"]["LastModified"][0]["last_update"],
                "%Y-%m-%dT%H:%M:%S.%f",
            )

        current_time = datetime.now()
        all_changed_data = data_preprocessor.load_preprocessed_data(
            current_time, last_updated
        )

        # identify names and add name-tags before chunking and translation
        all_changed_data = self.identify_names(all_changed_data)

        # transform long entries into multiple chunks and translation to english
        all_changed_data = transform_text_to_chunks(all_changed_data)
        print(
            self.weaviate_client.query.get("Embeddings", ["type_id", "text"])
            .do()
            .items()
        )
        if len(all_changed_data) != 0:
            ids = {data.id for data in all_changed_data}
            for type_id in ids:
                result = self.weaviate_client.batch.delete_objects(
                    "Embeddings",
                    where={
                        "path": ["type_id"],
                        "operator": "Equal",
                        "valueString": type_id,
                    },
                )
            self.vector_store.add_texts(
                [data.text for data in all_changed_data],
                [
                    {
                        "type_id": data.id,
                        "type": typ.value,
                        "last_changed": data.last_changed.isoformat(),
                        "text": data.text,
                    }
                    for data in all_changed_data
                ],
            )

            self.weaviate_client.data_object.create(
                {"type": typ.value, "last_update": current_time.isoformat()},
                "LastModified",
            )
            print(
                self.weaviate_client.query.get("Embeddings", ["type_id", "text"])
                .do()
                .items()
            )
            print(
                self.weaviate_client.query.get("LastModified", ["last_update", "type"])
                .do()
                .items()
            )

    def identify_names(self, all_data: List[DataInformation]) -> List[DataInformation]:
        """
        Method identifies names with spacy and adds name tags to the text
        :param all_data:  which is the List of DataInformation that gets send to the chunking
        :return: the input list with added name tags to persons
        """
        for data in all_data:
            # identify language of text
            language = self.get_target_language(data.text)
            # choose spacy model after language
            if language == "de":
                nlp = self.de_lang_nlp
            else:
                nlp = self.muulti_lang_nlp
            # identify sentence parts
            doc = nlp(data.text)
            already_replaced = []
            for ent in doc.ents:
                if ent.text in already_replaced or ent.label_ != "PER":
                    continue
                # only person names are flanked by tag and multiplicity is avoided
                already_replaced.append(ent.text)
                data.text = data.text.replace(ent.text, "<name>" + ent.text + "</name>")
        return all_data

    def get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def get_target_language(self, text):
        Language.factory("language_detector", func=self.get_lang_detector)
        if "sentencizer" not in self.muulti_lang_nlp.pipe_names:
            self.muulti_lang_nlp.add_pipe("sentencizer")
        if "language_detector" not in self.muulti_lang_nlp.pipe_names:
            self.muulti_lang_nlp.add_pipe("language_detector", last=True)
        doc = self.muulti_lang_nlp(text)
        if doc._.language["score"] > 0.8:
            return doc._.language["language"]
        else:
            return self.translator.translate_to(
                text, "EN-US"
            ).detected_source_lang.lower()
