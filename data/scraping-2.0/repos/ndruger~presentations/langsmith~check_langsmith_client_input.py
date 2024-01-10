from typing import Any, Optional
import datetime
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.chat_models import ChatOpenAI
import json


class TracerClient(Client):
    def create_run(
        self,
        name: str,
        inputs: dict[str, Any],
        run_type: str,
        *,
        execution_order: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        print(
            json.dumps(
                [
                    "create_run",
                    {
                        "type": "create_run",
                        "name": name,
                        "inputs": inputs,
                        "run_type": run_type,
                        "execution_order": execution_order,
                        "kwargs": kwargs,
                    },
                ],
                default=str,
                indent=2,
            )
        )

    def update_run(
        self,
        run_id: Any,
        *,
        end_time: Optional[datetime.datetime] = None,
        error: Optional[str] = None,
        inputs: Optional[dict] = None,
        outputs: Optional[dict] = None,
        events: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> None:
        print(
            json.dumps(
                [
                    "update_run",
                    {
                        "run_id": str(run_id),
                        "end_time": end_time,
                        "error": error,
                        "inputs": inputs,
                        "outputs": outputs,
                        "events": events,
                        "kwargs": kwargs,
                    },
                ],
                default=str,
                indent=2,
            )
        )


client = TracerClient()
tracer = LangChainTracer(use_threading=False, client=client)

llm = ChatOpenAI(callbacks=[tracer])
llm.invoke("Hello, world!")
