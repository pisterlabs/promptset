from typing import Any, Dict, List, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.open_ai_file_object import OpenAIFileObject
from ..models.open_ai_file_purpose import OpenAIFilePurpose
from ..models.open_ai_file_status import OpenAIFileStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="OpenAIFile")


@_attrs_define
class OpenAIFile:
    """The `File` object represents a document that has been uploaded to OpenAI.

    Attributes:
        id (str): The file identifier, which can be referenced in the API endpoints.
        bytes_ (int): The size of the file, in bytes.
        created_at (int): The Unix timestamp (in seconds) for when the file was created.
        filename (str): The name of the file.
        object_ (OpenAIFileObject): The object type, which is always `file`.
        purpose (OpenAIFilePurpose): The intended purpose of the file. Supported values are `fine-tune`, `fine-tune-
            results`, `assistants`, and `assistants_output`.
        status (OpenAIFileStatus): Deprecated. The current status of the file, which can be either `uploaded`,
            `processed`, or `error`.
        status_details (Union[Unset, str]): Deprecated. For details on why a fine-tuning training file failed
            validation, see the `error` field on `fine_tuning.job`.
    """

    id: str
    bytes_: int
    created_at: int
    filename: str
    object_: OpenAIFileObject
    purpose: OpenAIFilePurpose
    status: OpenAIFileStatus
    status_details: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        bytes_ = self.bytes_
        created_at = self.created_at
        filename = self.filename
        object_ = self.object_.value

        purpose = self.purpose.value

        status = self.status.value

        status_details = self.status_details

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "bytes": bytes_,
                "created_at": created_at,
                "filename": filename,
                "object": object_,
                "purpose": purpose,
                "status": status,
            }
        )
        if status_details is not UNSET:
            field_dict["status_details"] = status_details

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        bytes_ = d.pop("bytes")

        created_at = d.pop("created_at")

        filename = d.pop("filename")

        object_ = OpenAIFileObject(d.pop("object"))

        purpose = OpenAIFilePurpose(d.pop("purpose"))

        status = OpenAIFileStatus(d.pop("status"))

        status_details = d.pop("status_details", UNSET)

        open_ai_file = cls(
            id=id,
            bytes_=bytes_,
            created_at=created_at,
            filename=filename,
            object_=object_,
            purpose=purpose,
            status=status,
            status_details=status_details,
        )

        open_ai_file.additional_properties = d
        return open_ai_file

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
