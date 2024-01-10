import asyncio
import concurrent
import copy
import inspect
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, cast

from gentrace.api_client import ApiClient
from gentrace.apis.tags.v1_api import V1Api
from gentrace.configuration import Configuration
from gentrace.providers.context import Context
from gentrace.providers.pipeline import Pipeline
from gentrace.providers.step_run import StepRun
from gentrace.providers.utils import (
    from_date_string,
    get_test_counter,
    is_openai_v1,
    run_post_background,
    to_date_string,
)

_pipeline_run_loop = None
_pipeline_tasks = []


# https://stackoverflow.com/a/63110035/1057411
def fire_and_forget(coro):
    global _pipeline_run_loop, _pipeline_tasks
    if _pipeline_run_loop is None:
        _pipeline_run_loop = asyncio.new_event_loop()
        threading.Thread(target=_pipeline_run_loop.run_forever, daemon=True).start()
    if inspect.iscoroutine(coro):
        task = asyncio.run_coroutine_threadsafe(coro, _pipeline_run_loop)
        _pipeline_tasks.append(task)


def flush():
    global _pipeline_tasks
    if _pipeline_tasks:
        # Wait for all tasks to complete
        concurrent.futures.wait(_pipeline_tasks)
        _pipeline_tasks.clear()


class PipelineRun:
    def __init__(
            self, pipeline, id: Optional[str] = None, context: Optional[Context] = None
    ):
        self.pipeline: Pipeline = pipeline
        self.pipeline_run_id: str = id or str(uuid.uuid4())
        self.step_runs: List[StepRun] = []
        self.context: Context = context or {}

    def get_id(self):
        return self.pipeline_run_id

    def get_pipeline(self):
        return self.pipeline

    def get_openai(self, asynchronous=False):
        if "openai" in self.pipeline.pipeline_handlers:

            if is_openai_v1():
                from gentrace.providers.llms.openai_v1 import (
                    GentraceAsyncOpenAI,
                    GentraceSyncOpenAI,
                )

                if asynchronous:
                    openai_async_handler = GentraceAsyncOpenAI(**self.pipeline.openai_config,
                                                               gentrace_config=self.pipeline.config,
                                                               pipeline=self, pipeline_run=self)

                    from openai import AsyncOpenAI

                    typed_cloned_handler = cast(AsyncOpenAI, openai_async_handler)
                    return typed_cloned_handler
                else:
                    openai_handler = GentraceSyncOpenAI(**self.pipeline.openai_config,
                                                        gentrace_config=self.pipeline.config,
                                                        pipeline=self, pipeline_run=self)

                    from openai import OpenAI

                    typed_cloned_handler = cast(OpenAI, openai_handler)
                    return typed_cloned_handler
            else:
                handler = self.pipeline.pipeline_handlers.get("openai")
                cloned_handler = copy.deepcopy(handler)

                from .llms.openai_v0 import annotate_pipeline_handler

                annotated_handler = annotate_pipeline_handler(
                    cloned_handler, self.pipeline.openai_config, self
                )

                import openai

                # TODO: Could not find an easy way to create a union type with openai and
                # OpenAIPipelineHandler, so we just use openai.
                typed_cloned_handler = cast(openai, annotated_handler)

                return typed_cloned_handler
        else:
            raise ValueError(
                "Did not find OpenAI handler. Did you call setup() on the pipeline?"
            )

    def get_pinecone(self):
        if "pinecone" in self.pipeline.pipeline_handlers:
            handler = self.pipeline.pipeline_handlers.get("pinecone")
            cloned_handler = copy.deepcopy(handler)
            cloned_handler.set_pipeline_run(self)
            import pinecone

            return cast(pinecone, cloned_handler)
        else:
            raise ValueError(
                "Did not find Pinecone handler. Did you call setup() on the pipeline?"
            )

    def add_step_run(self, step_run: StepRun):
        self.step_runs.append(step_run)

    async def ameasure(self, func: Callable[..., Any], **kwargs):
        """
        Asynchronously measures the execution time of a function and logs the result as a `StepRun`.
        Also logs additional information about the function invocation.

        Parameters:
        func (Callable[..., Any]): The asynchronous function whose execution time is to be measured.
        **kwargs: Arbitrary keyword arguments. These are passed directly to the function.
                  If a "step_info" argument is included, it should be a dictionary containing
                  additional metadata about the function invocation. Supported keys are "provider",
                  "invocation", and "model_params". The "step_info" argument is not passed to the
                  function.

        Returns:
        The return value of the function invocation.

        Raises:
        Any exceptions raised by the function will be propagated.

        Example:
        async def add(x, y):
            return x + y

        await measure(add, x=1, y=2, step_info={"provider": "my_provider", "invocation": "add invocation"})
        """
        input_params = {k: v for k, v in kwargs.items() if k not in ["step_info"]}

        step_info = kwargs.get("step_info", {})

        start_time = time.time()
        output = func(**input_params)
        end_time = time.time()

        elapsed_time = end_time - start_time

        self.add_step_run(
            StepRun(
                step_info.get("provider", "undeclared"),
                step_info.get("invocation", "undeclared"),
                elapsed_time,
                start_time,
                end_time,
                input_params,
                step_info.get("model_params", {}),
                output,
                step_info.get("context", {}),
            )
        )

    def measure(self, func: Callable[..., Any], **kwargs):
        """
        Measures the execution time of a function and logs the result as a `StepRun`.
        Also logs additional information about the function invocation.

        Parameters:
        func (Callable[..., Any]): The function whose execution time is to be measured.
        **kwargs: Arbitrary keyword arguments. These are passed directly to the function.
                  If a "step_info" argument is included, it should be a dictionary containing
                  additional metadata about the function invocation. Supported keys are "provider",
                  "invocation", and "model_params". The "step_info" argument is not passed to the
                  function.

        Returns:
        The return value of the function invocation.

        Raises:
        Any exceptions raised by the function will be propagated.

        Example:
        def add(x, y):
            return x + y

        measure(add, x=1, y=2, step_info={"provider": "my_provider", "invocation": "add invocation"})
        """
        input_params = {k: v for k, v in kwargs.items() if k not in ["step_info"]}

        step_info = kwargs.get("step_info", {})

        start_time = time.time()
        outputs = func(**input_params)
        end_time = time.time()

        outputs_for_step_run = outputs

        if not isinstance(outputs_for_step_run, dict):
            outputs_for_step_run = {"value": outputs_for_step_run}

        elapsed_time = int(end_time - start_time)

        self.add_step_run(
            StepRun(
                step_info.get("provider", "undeclared"),
                step_info.get("invocation", "undeclared"),
                elapsed_time,
                to_date_string(start_time),
                to_date_string(end_time),
                input_params,
                step_info.get("model_params", {}),
                outputs_for_step_run,
                step_info.get("context", {}),
            )
        )

        return outputs

    def checkpoint(self, step_info):
        """
        Creates a checkpoint by recording a `StepRun` instance with execution metadata and appending it to `self.step_runs`.
        If there are no prior steps, elapsed time is set to 0 and start and end times are set to the current timestamp.
        If prior steps exist, calculates the elapsed time using the end time of the last `StepRun`.

        Parameters:
        step_info (dict): The information about the step to checkpoint. It should include "inputs" and "outputs".
                    Optionally, it can also include "provider", "invocation" and "modelParams".

        Returns:
        None

        Example:
        checkpoint_step = {
            "provider": "MyProvider",
            "invocation": "doSomething",
            "inputs": {"x": 10, "y": 20},
            "outputs": {"result": 30}
        }

        checkpoint(checkpoint_step)
        """
        last_element = self.step_runs[-1] if self.step_runs else None

        if last_element:
            step_start_time = from_date_string(last_element.end_time)
            end_time_new = time.time()
            elapsed_time = int(end_time_new - step_start_time)
            self.step_runs.append(
                StepRun(
                    step_info.get("provider", "undeclared"),
                    step_info.get("invocation", "undeclared"),
                    elapsed_time,
                    to_date_string(step_start_time),
                    to_date_string(end_time_new),
                    step_info.get("inputs", {}),
                    step_info.get("modelParams", {}),
                    step_info.get("outputs", {}),
                    step_info.get("context", {}),
                )
            )
        else:
            elapsed_time = 0
            start_and_end_time = time.time()
            self.step_runs.append(
                StepRun(
                    step_info.get("provider", "undeclared"),
                    step_info.get("invocation", "undeclared"),
                    elapsed_time,
                    to_date_string(start_and_end_time),
                    to_date_string(start_and_end_time),
                    step_info.get("inputs", {}),
                    step_info.get("modelParams", {}),
                    step_info.get("outputs", {}),
                    step_info.get("context", {}),
                )
            )

    async def asubmit(self) -> Dict:
        if get_test_counter() > 0:
            return {
                "pipelineRunId": self.get_id(),
            }

        configuration = Configuration(host=self.pipeline.config.get("host"))
        configuration.access_token = self.pipeline.config.get("api_key")
        api_client = ApiClient(configuration=configuration)
        v1_api = V1Api(api_client=api_client)

        step_runs_data = [
            {
                "providerName": step_run.provider,
                "invocation": step_run.invocation,
                "modelParams": step_run.model_params,
                "inputs": step_run.inputs,
                "outputs": step_run.outputs,
                "elapsedTime": step_run.elapsed_time,
                "startTime": step_run.start_time,
                "endTime": step_run.end_time,
                "context": {**self.context, **step_run.context},
            }
            for step_run in self.step_runs
        ]

        try:
            pipeline_post_response = await run_post_background(
                v1_api,
                {
                    "id": self.pipeline_run_id,
                    "slug": self.pipeline.slug,
                    "stepRuns": step_runs_data,
                },
            )
            return {
                "pipelineRunId": pipeline_post_response.body.get_item_oapg(
                    "pipelineRunId"
                )
            }
        except Exception as e:
            print(f"Error submitting to Gentrace: {e}")
            return {"pipelineRunId": None}

    def submit(self, wait_for_server=False) -> Dict:
        if get_test_counter() > 0:
            return {
                "pipelineRunId": self.get_id(),
            }

        configuration = Configuration(host=self.pipeline.config.get("host"))
        configuration.access_token = self.pipeline.config.get("api_key")

        api_client = ApiClient(configuration=configuration)
        v1_api = V1Api(api_client=api_client)

        merged_metadata = {}

        step_runs_data = []
        for step_run in self.step_runs:
            # Extract metadata without mutating original contexts
            this_context = copy.deepcopy(self.context)
            this_context_metadata = this_context.get("metadata", {})
            step_run_context = copy.deepcopy(step_run.context)
            step_run_context_metadata = step_run_context.get("metadata", {})

            merged_metadata.update(this_context_metadata)
            merged_metadata.update(step_run_context_metadata)

            this_context.pop("metadata", None)
            step_run_context.pop("metadata", None)

            this_context.pop("previousRunId", None)
            step_run_context.pop("previousRunId", None)

            step_runs_data.append(
                {
                    "providerName": step_run.provider,
                    "invocation": step_run.invocation,
                    "modelParams": step_run.model_params,
                    "inputs": step_run.inputs,
                    "outputs": step_run.outputs,
                    "elapsedTime": step_run.elapsed_time,
                    "startTime": step_run.start_time,
                    "endTime": step_run.end_time,
                    "context": {**this_context, **step_run_context},
                }
            )

        if len(step_runs_data) == 0:
            return {"pipelineRunId": None}

        if not wait_for_server:
            fire_and_forget(
                run_post_background(
                    v1_api,
                    {
                        "id": self.pipeline_run_id,
                        "slug": self.pipeline.slug,
                        "metadata": merged_metadata,
                        "previousRunId": self.context.get("previousRunId"),
                        "stepRuns": step_runs_data,
                    },
                )
            )

            return {"pipelineRunId": self.pipeline_run_id}

        if wait_for_server:
            try:
                pipeline_post_response = v1_api.v1_run_post(
                    {
                        "id": self.pipeline_run_id,
                        "slug": self.pipeline.slug,
                        "metadata": merged_metadata,
                        "previousRunId": self.context.get("previousRunId"),
                        "stepRuns": step_runs_data,
                    }
                )
                return {
                    "pipelineRunId": pipeline_post_response.body.get_item_oapg(
                        "pipelineRunId"
                    )
                }

            except Exception as e:
                print(f"Error submitting to Gentrace: {e}")
                return {"pipelineRunId": None}
