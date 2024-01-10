import os
import typing as t

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from singer_sdk import exceptions
from singer_sdk import typing as th
from singer_sdk._singerlib.messages import Message, SchemaMessage

from map_gpt_embeddings.sdk_fixes.mapper_base import BasicPassthroughMapper
from map_gpt_embeddings.sdk_fixes.messages import RecordMessage
from map_gpt_embeddings.stream import OpenAIStream
from map_gpt_embeddings.tap import TapOpenAI


class GPTEmbeddingMapper(BasicPassthroughMapper):
    """Split documents into segments, then vectorize."""

    name = "map-gpt-embeddings"

    def __init__(self, *args, **kwargs):
        """Initialize the mapper.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.tap = TapOpenAI(config=dict(self.config))
        self.stream = None

    def map_schema_message(self, message_dict: dict) -> t.Iterable[Message]:
        for result in t.cast(
            t.Iterable[SchemaMessage], super().map_schema_message(message_dict)
        ):
            # Add an "embeddings" property to the schema
            result.schema["properties"]["embeddings"] = th.ArrayType(
                th.NumberType
            ).to_dict()
            self.stream = OpenAIStream(tap=self.tap, schema=result.schema)
            yield result

    config_jsonschema = th.PropertiesList(
        th.Property(
            "document_text_property",
            th.StringType,
            default="page_content",
            description="The name of the property containing the document text."
        ),
        th.Property(
            "document_metadata_property",
            th.StringType,
            description="The name of the property containing the document metadata."
        ),
        th.Property(
            "openai_api_key",
            th.StringType,
            secret=True,
            description="OpenAI API key. Optional if `OPENAI_API_KEY` env var is set.",
        ),
        th.Property(
            "msi_client_id",
            th.StringType,
            description="Azure Managed Identity for authentication"
        ),
        th.Property(
            "use_msi",
            th.BooleanType,
            description="Use Azure Managed Identity for authentication",
            default=False
        ),
        th.Property(
            "api_endpoint",
            th.StringType,
            description="Azure OpenAI API Endpoint",
            default="https://api.openai.com"
        ),
        th.Property(
            "deployment_name",
            th.StringType,
            description="Azure OpenAI Deployment Name"
        ),
    ).to_dict()

    def _validate_config(
        self,
        *,
        raise_errors: bool = True,
        warnings_as_errors: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Validate configuration input against the plugin configuration JSON schema.

        Args:
            raise_errors: Flag to throw an exception if any validation errors are found.
            warnings_as_errors: Flag to throw an exception if any warnings were emitted.

        Returns:
            A tuple of configuration validation warnings and errors.

        Raises:
            ConfigValidationError: If raise_errors is True and validation fails.
        """
        warnings, errors = super()._validate_config(
            raise_errors=raise_errors, warnings_as_errors=warnings_as_errors
        )
        if (
            raise_errors
            and self.config.get("openai_api_key", None) is None
            and "OPENAI_API_KEY" not in os.environ
            and self.config.get("use_msi", None) is None
        ):
            raise exceptions.ConfigValidationError(
                "Must set at least one of the following: `openai_api_key` setting, "
                "`use_msi` to true, "
                f"`{self.name.upper().replace('-', '_')}_OPEN_API_KEY` env var, or "
                " `OPENAI_API_KEY` env var."
            )
        return warnings, errors

    def split_record(self, record: dict) -> t.Iterable[dict]:
        """Split a record dict to zero or more record dicts.

        Args:
            record: The record object to split.

        Yields:
            A generator of record dicts.
        """
        raw_document_text = record[self.config["document_text_property"]]

        if self.config.get("document_metadata_property", None):
            metadata_dict = record[self.config["document_metadata_property"]]
        else:
            metadata_dict = {}

        if not self.config.get("split_documents", True):
            yield record
            return

        splitter_config = self.config.get("splitter_config", {})
        if "chunk_size" not in splitter_config:
            splitter_config["chunk_size"] = 1000
        if "chunk_overlap" not in splitter_config:
            splitter_config["chunk_overlap"] = 200
        text_splitter = RecursiveCharacterTextSplitter(**splitter_config)

        document = Document(page_content=raw_document_text, metadata=metadata_dict)

        document_segments = text_splitter.split_documents([document])

        # assert document_segments and len(
        #     document_segments
        # ), "No documents output from split."
        if len(document_segments) > 1:
            self.logger.debug("Document split into %s segments", len(document_segments))
        elif len(document_segments) == 1:
            self.logger.debug("Document not split", len(document_segments))

        for doc_segment in document_segments:
            new_record = record.copy()
            new_record[self.config["document_text_property"]] = doc_segment.page_content
            if self.config.get("document_metadata_property", None):
                new_record[self.config.get["document_metadata_property"]] = doc_segment.metadata
            yield new_record

    def map_record_message(self, message_dict: dict) -> t.Iterable[RecordMessage]:
        for split_record in self.split_record(message_dict["record"]):
            response_data = list(
                self.stream.request_records(
                    {
                        "text": split_record[self.config["document_text_property"]],
                        "model": "text-embedding-ada-002",
                    }
                )
            )
            split_record["embeddings"] = response_data[0]["data"][0]["embedding"]
            new_message = message_dict.copy()
            new_message["record"] = split_record

            yield t.cast(RecordMessage, RecordMessage.from_dict(new_message))


if __name__ == "__main__":
    GPTEmbeddingMapper.cli()
