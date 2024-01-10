from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.patchee import Patchee

from . import _SPAN_NAME_FINE_TUNNING, OpenaiPatchees


class _FineTuningPatchees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None], method_name: str):
        fine_tuning_jobs = Patchee(
            obj=client.fine_tuning.jobs if client else openai.fine_tuning.jobs,
            method_name=method_name,
            span_name=_SPAN_NAME_FINE_TUNNING,
            event_name=f"fine_tuning.jobs.{method_name}",
        )

        fine_tuning_raw_jobs = Patchee(
            obj=client.fine_tuning.with_raw_response.jobs
            if client
            else openai.fine_tuning.with_raw_response.jobs,
            method_name=method_name,
            span_name=_SPAN_NAME_FINE_TUNNING,
            event_name=f"fine_tuning.with_raw_response.jobs.{method_name}",
        )

        fine_tuning_jobs_raw = Patchee(
            obj=client.fine_tuning.jobs.with_raw_response
            if client
            else openai.fine_tuning.jobs.with_raw_response,
            method_name=method_name,
            span_name=_SPAN_NAME_FINE_TUNNING,
            event_name=f"fine_tuning.jobs.with_raw_response.{method_name}",
        )

        self.patchees = [
            fine_tuning_jobs,
            fine_tuning_raw_jobs,
            fine_tuning_jobs_raw,
        ]

        if client:
            raw_fine_tuning = Patchee(
                obj=client.with_raw_response.fine_tuning.jobs,
                method_name=method_name,
                span_name=_SPAN_NAME_FINE_TUNNING,
                event_name=f"with_raw_response.fine_tuning.jobs.{method_name}",
            )
            self.patchees.append(raw_fine_tuning)


class _FineTuningListPatchees(_FineTuningPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list")


class _FineTuningCreatePatchees(_FineTuningPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="create")


class _FineTuningCancelPatchees(_FineTuningPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="cancel")


class _FineTuningRetrievePatchees(_FineTuningPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="retrieve")


class _FineTuningListEventsPatchees(_FineTuningPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        super().__init__(client=client, method_name="list_events")


class FineTuningPatchees(OpenaiPatchees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        list = _FineTuningListPatchees(client=client)
        create = _FineTuningCreatePatchees(client=client)
        cancel = _FineTuningCancelPatchees(client=client)
        retrieve = _FineTuningRetrievePatchees(client=client)
        list_events = _FineTuningListEventsPatchees(client=client)

        patchees = (
            list.patchees
            + create.patchees
            + cancel.patchees
            + retrieve.patchees
            + list_events.patchees
        )

        super().__init__(patchees=patchees, client=client)
