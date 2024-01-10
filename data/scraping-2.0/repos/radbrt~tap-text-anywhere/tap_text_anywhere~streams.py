"""Stream type classes for tap-text-anywhere."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Generator, Iterable

import textract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from singer_sdk import typing as th  # JSON Schema typing helpers

from tap_text_anywhere.client import text_anywhereStream
from hashlib import md5


class TextStream(text_anywhereStream):
    """Define custom stream."""
    
    primary_keys = ["filename", "part_number"]
    replication_key = "updated_at"
    # Optionally, you may also use `schema_filepath` in place of `schema`:
    # schema_filepath = SCHEMAS_DIR / "users.json"  # noqa: ERA001
    schema = th.PropertiesList(
        th.Property("name", th.StringType),
        th.Property(
            "filename",
            th.StringType,
            description="The name of the file",
        ),
        th.Property(
            "textcontent",
            th.StringType,
        ),
        th.Property(
            "updated_at",
            th.StringType,
            description="The last time the file was updated",
        ),
        th.Property("hash", th.StringType, description="The hash of the chunk"),
        th.Property("part_number", th.IntegerType, description="chunk number")
    ).to_dict()

    def get_records(
        self,
        context: dict | None,
    ) -> Iterable[dict]:
        """Return a generator of record-type dictionary objects.

        The optional `context` argument is used to identify a specific slice of the
        stream if partitioning is required for the stream. Most implementations do not
        require partitioning and should ignore the `context` argument.

        Args:
            context: Stream partition or context dictionary.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=self.config.get("chunk_size", 2000),
            chunk_overlap=self.config.get("chunk_overlap", 500),
            length_function=len,
        )

        empty = True

        for file in self.filesystem.ls(self.config["filepath"], detail=True):
            # RegEx is currently checked against basename rather than full path.
            # Fullpath matching could be added if recursive subdirectory syncing is
            # implemented.
            if file["type"] == "directory" or file["size"] == 0:
                continue
            if "file_regex" in self.config and not re.match(
                self.config["file_regex"],
                Path(file["name"]).name,
            ):
                continue
            empty = False

            filename = file["name"].split("/")[-1]

            last_modified = file.get("LastModified")
            updated_at = last_modified and last_modified.isoformat()

            if self.config["protocol"] == "s3":
                if file["LastModified"].isoformat() < self.config["start_date"] or file[
                    "LastModified"
                ].isoformat() <= self.get_starting_replication_key_value(context):
                    continue

                tmpfile = tempfile.NamedTemporaryFile(
                    suffix=filename,
                    delete=False,
                ).name

                self.filesystem.get(file["name"], tmpfile)

                text = textract.process(tmpfile)
                chunks = text_splitter.create_documents([text.decode("utf-8")])
                for i, chunk in enumerate(chunks):
                    yield {
                        "filename": filename,
                        "textcontent": chunk.page_content,
                        "updated_at": updated_at,
                        "part_number": i,
                        "hash": md5(chunk.page_content.encode("utf-8")).hexdigest(),
                    }

            elif self.config["protocol"] == "file":
                text = textract.process(file["name"])
                chunks = text_splitter.create_documents([text.decode("utf-8")])
                for i, chunk in enumerate(chunks):
                    yield {
                        "filename": filename,
                        "textcontent": chunk.page_content,
                        "updated_at": updated_at,
                        "part_number": i,
                        "hash": md5(chunk.page_content.encode("utf-8")).hexdigest(),
                    }
            else:
                msg = "Don't know that protocol yet"
                raise NotImplementedError(msg)

        if empty:
            msg = (
                "No files found. Choose a different `filepath` or try a more lenient "
                "`file_regex`."
            )
            raise RuntimeError(msg)

        # for file in self.get_files():
        #     for chunk in chunks:

    def get_files(self) -> Generator[str, None, None]:
        """Gets file names to be synced.

        Yields:
            The name of a file to be synced, matching a regex pattern, if one has been
                configured.
        """
        empty = True

        for file in self.filesystem.ls(self.config["filepath"], detail=True):
            # RegEx is currently checked against basename rather than full path.
            # Fullpath matching could be added if recursive subdirectory syncing is
            # implemented.
            if file["type"] == "directory" or file["size"] == 0:
                continue
            if "file_regex" in self.config and not re.match(
                self.config["file_regex"],
                Path(file["name"]).name,
            ):
                continue
            empty = False
            if self.config["protocol"] == "s3":
                if (
                    file["LastModified"] < self.config["start_date"]
                    or file["LastModified"] < self.get_starting_replication_key_value()
                ):
                    continue
                filename = file["name"].split("/")[-1]
                tmpfile = tempfile.NamedTemporaryFile(
                    suffix=filename,
                    delete=False,
                ).name

                self.logger.info(f"Downloading {filename} to {tmpfile}")
                self.logger.info(f"Full file name: {file['name']}")

                self.filesystem.get(file["name"], tmpfile)
                yield tmpfile
            elif self.config["protocol"] == "file":
                yield file["name"]
            else:
                msg = "Don't know that protocol yet"
                raise NotImplementedError(msg)

        if empty:
            msg = (
                "No files found. Choose a different `filepath` or try a more lenient "
                "`file_regex`."
            )
            raise RuntimeError(msg)
