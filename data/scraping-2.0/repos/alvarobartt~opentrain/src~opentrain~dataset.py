import json
import warnings
from functools import cached_property
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Union
from uuid import uuid4

import openai
from openai.error import TryAgain

FILE_SIZE_WARNING = 500 * 1024 * 1024


class Dataset:
    """The `Dataset` class is not just a wrapper around OpenAI's File API, but it also
    provides some useful methods to work with datasets.

    Args:
        file_id: the ID of the file previously uploaded to OpenAI.
        organization: the OpenAI organization name. Defaults to None.

    Attributes:
        file_id: the ID of the file previously uploaded to OpenAI.
        organization: the OpenAI organization name.
        info: the information of the file.

    Examples:
        >>> from opentrain import Dataset
        >>> dataset = Dataset(file_id="file-1234")
        >>> dataset.info
        >>> content = dataset.download()
        >>> dataset.delete()
    """

    def __init__(self, file_id: str, organization: Union[str, None] = None) -> None:
        """Initializes the `Dataset` class.

        Args:
            file_id: the ID of the file previously uploaded to OpenAI.
            organization: the OpenAI organization name. Defaults to None.
        """
        self.file_id = file_id
        self.organization = organization

    @cached_property
    def info(self) -> Dict[str, Any]:
        """Returns the information of the file uploaded to OpenAI.

        Returns:
            A dictionary with the information of the file.
        """
        return openai.File.retrieve(id=self.file_id, organization=self.organization)

    def download(self) -> bytes:
        """Downloads the file from OpenAI.

        Returns:
            The content of the file as bytes.
        """
        warnings.warn(
            "Dataset.download() is just available for paid/pro accounts, so bear in"
            " mind that this will fail if you're using a free tier.",
            stacklevel=2,
        )
        return openai.File.download(id=self.file_id, organization=self.organization)

    def to_file(self, output_path: str) -> None:
        """Downloads the file from OpenAI and saves it to the specified path.

        Args:
            output_path: the path where the file will be saved.
        """
        content = self.download()
        with open(output_path, "wb") as f:
            f.write(content)
        del content

    def delete(self) -> None:
        """Deletes the file from OpenAI."""
        file_deleted = False
        while file_deleted is False:
            try:
                openai.File.delete(
                    sid=self.file_id, organization=self.organization, request_timeout=10
                )
                file_deleted = True
            except TryAgain:
                sleep(1)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        file_name: Union[str, None] = None,
        organization: Union[str, None] = None,
    ) -> "Dataset":
        """Uploads a file to OpenAI and returns a `Dataset` object.

        Args:
            file_path: the path of the file to be uploaded.
            file_name: the name of the file to be defined in OpenAI. Defaults to None.
            organization: the OpenAI organization name. Defaults to None.

        Returns:
            A `Dataset` object.
        """
        upload_response = openai.File.create(
            file=open(file_path, "rb"),
            organization=organization,
            purpose="fine-tune",
            user_provided_filename=file_name,
        )
        return cls(file_id=upload_response.id, organization=organization)

    @classmethod
    def from_records(
        cls,
        records: List[Dict[str, str]],
        file_name: Union[str, None] = None,
        organization: Union[str, None] = None,
    ) -> "Dataset":
        """Uploads a list of records to OpenAI and returns a `Dataset` object. Note
        that this function saves it first to a local file and then uploads it to
        OpenAI.

        Args:
            records: a list of dictionaries with the records to be uploaded.
            file_name: the name of the file to be defined in OpenAI. Defaults to None.
            organization: the OpenAI organization name. Defaults to None.

        Returns:
            A `Dataset` object.
        """
        local_path = (
            Path.home() / ".cache" / "opentrain" / f"{file_name or uuid4()}.jsonl"
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path.as_posix(), "w") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

        if local_path.stat().st_size > FILE_SIZE_WARNING:
            warnings.warn(
                f"Your file is larger than {FILE_SIZE_WARNING / 1024 / 1024} MB, and"
                " the maximum total upload file size in OpenAI is 1GB, so please be"
                " aware that if you already have uploaded files to OpenAI this might"
                " fail. If you need to upload larger files or require more space,"
                " please contact OpenAI as suggested at"
                " https://platform.openai.com/docs/api-reference/files/upload.",
                stacklevel=2,
            )

        upload_response = openai.File.create(
            file=open(local_path.as_posix(), "rb"),
            organization=organization,
            purpose="fine-tune",
            user_provided_filename=file_name,
        )
        return cls(file_id=upload_response.id, organization=organization)


class File(Dataset):
    """This class is just a wrapper around `Dataset` with the same functionality. It's
    just here to keep the same naming convention as OpenAI."""

    pass


def list_datasets(organization: Union[str, None] = None) -> List[Dataset]:
    """Lists the datasets uploaded to your OpenAI or your organization's account.

    Args:
        organization: the OpenAI organization name. Defaults to None.

    Returns:
        A list of `Dataset` objects.
    """
    return [
        Dataset(file_id=file["id"], organization=organization)
        for file in openai.File.list(organization=organization)["data"]
    ]


def list_files(organization: Union[str, None] = None) -> List[File]:
    """This function is just a wrapper around `list_datasets` with the same
    functionality. It's just here to keep the same naming convention as OpenAI.

    Args:
        organization: the OpenAI organization name. Defaults to None.

    Returns:
        A list of `File` objects.
    """
    return [
        File(file_id=file["id"], organization=organization)
        for file in openai.File.list(organization=organization)["data"]
    ]
