from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.fine_tune_object import FineTuneObject
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.fine_tune_event import FineTuneEvent
    from ..models.fine_tune_hyperparams import FineTuneHyperparams
    from ..models.open_ai_file import OpenAIFile


T = TypeVar("T", bound="FineTune")


@_attrs_define
class FineTune:
    """The `FineTune` object represents a legacy fine-tune job that has been created through the API.

    Attributes:
        id (str): The object identifier, which can be referenced in the API endpoints.
        created_at (int): The Unix timestamp (in seconds) for when the fine-tuning job was created.
        hyperparams (FineTuneHyperparams): The hyperparameters used for the fine-tuning job. See the [fine-tuning
            guide](/docs/guides/legacy-fine-tuning/hyperparameters) for more details.
        model (str): The base model that is being fine-tuned.
        object_ (FineTuneObject): The object type, which is always "fine-tune".
        organization_id (str): The organization that owns the fine-tuning job.
        result_files (List['OpenAIFile']): The compiled results files for the fine-tuning job.
        status (str): The current status of the fine-tuning job, which can be either `created`, `running`, `succeeded`,
            `failed`, or `cancelled`.
        training_files (List['OpenAIFile']): The list of files used for training.
        updated_at (int): The Unix timestamp (in seconds) for when the fine-tuning job was last updated.
        validation_files (List['OpenAIFile']): The list of files used for validation.
        events (Union[Unset, List['FineTuneEvent']]): The list of events that have been observed in the lifecycle of the
            FineTune job.
        fine_tuned_model (Optional[str]): The name of the fine-tuned model that is being created.
    """

    id: str
    created_at: int
    hyperparams: "FineTuneHyperparams"
    model: str
    object_: FineTuneObject
    organization_id: str
    result_files: List["OpenAIFile"]
    status: str
    training_files: List["OpenAIFile"]
    updated_at: int
    validation_files: List["OpenAIFile"]
    fine_tuned_model: Optional[str]
    events: Union[Unset, List["FineTuneEvent"]] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        created_at = self.created_at
        hyperparams = self.hyperparams.to_dict()

        model = self.model
        object_ = self.object_.value

        organization_id = self.organization_id
        result_files = []
        for result_files_item_data in self.result_files:
            result_files_item = result_files_item_data.to_dict()

            result_files.append(result_files_item)

        status = self.status
        training_files = []
        for training_files_item_data in self.training_files:
            training_files_item = training_files_item_data.to_dict()

            training_files.append(training_files_item)

        updated_at = self.updated_at
        validation_files = []
        for validation_files_item_data in self.validation_files:
            validation_files_item = validation_files_item_data.to_dict()

            validation_files.append(validation_files_item)

        events: Union[Unset, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.events, Unset):
            events = []
            for events_item_data in self.events:
                events_item = events_item_data.to_dict()

                events.append(events_item)

        fine_tuned_model = self.fine_tuned_model

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "hyperparams": hyperparams,
                "model": model,
                "object": object_,
                "organization_id": organization_id,
                "result_files": result_files,
                "status": status,
                "training_files": training_files,
                "updated_at": updated_at,
                "validation_files": validation_files,
                "fine_tuned_model": fine_tuned_model,
            }
        )
        if events is not UNSET:
            field_dict["events"] = events

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.fine_tune_event import FineTuneEvent
        from ..models.fine_tune_hyperparams import FineTuneHyperparams
        from ..models.open_ai_file import OpenAIFile

        d = src_dict.copy()
        id = d.pop("id")

        created_at = d.pop("created_at")

        hyperparams = FineTuneHyperparams.from_dict(d.pop("hyperparams"))

        model = d.pop("model")

        object_ = FineTuneObject(d.pop("object"))

        organization_id = d.pop("organization_id")

        result_files = []
        _result_files = d.pop("result_files")
        for result_files_item_data in _result_files:
            result_files_item = OpenAIFile.from_dict(result_files_item_data)

            result_files.append(result_files_item)

        status = d.pop("status")

        training_files = []
        _training_files = d.pop("training_files")
        for training_files_item_data in _training_files:
            training_files_item = OpenAIFile.from_dict(training_files_item_data)

            training_files.append(training_files_item)

        updated_at = d.pop("updated_at")

        validation_files = []
        _validation_files = d.pop("validation_files")
        for validation_files_item_data in _validation_files:
            validation_files_item = OpenAIFile.from_dict(validation_files_item_data)

            validation_files.append(validation_files_item)

        events = []
        _events = d.pop("events", UNSET)
        for events_item_data in _events or []:
            events_item = FineTuneEvent.from_dict(events_item_data)

            events.append(events_item)

        fine_tuned_model = d.pop("fine_tuned_model")

        fine_tune = cls(
            id=id,
            created_at=created_at,
            hyperparams=hyperparams,
            model=model,
            object_=object_,
            organization_id=organization_id,
            result_files=result_files,
            status=status,
            training_files=training_files,
            updated_at=updated_at,
            validation_files=validation_files,
            events=events,
            fine_tuned_model=fine_tuned_model,
        )

        fine_tune.additional_properties = d
        return fine_tune

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
