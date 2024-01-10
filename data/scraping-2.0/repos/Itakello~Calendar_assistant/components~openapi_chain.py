from __future__ import annotations

from typing import Any, Dict, Optional, cast

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import OpenAPIEndpointChain
from requests import Response


class OpenAPIEndpointChainRaw(OpenAPIEndpointChain):
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        intermediate_steps = {}
        instructions = inputs[self.instructions_key]
        instructions = instructions[: self.max_text_length]
        _api_arguments = self.api_request_chain.predict_and_parse(
            instructions=instructions, callbacks=_run_manager.get_child()
        )
        api_arguments = cast(str, _api_arguments)
        intermediate_steps["request_args"] = api_arguments
        _run_manager.on_text(
            api_arguments, color="green", end="\n", verbose=self.verbose
        )
        if api_arguments.startswith("ERROR"):
            return self._get_output(api_arguments, intermediate_steps)
        elif api_arguments.startswith("MESSAGE:"):
            return self._get_output(
                api_arguments[len("MESSAGE:") :], intermediate_steps
            )
        try:
            request_args = self.deserialize_json_input(api_arguments)
            method = getattr(self.requests, self.api_operation.method.value)
            api_response: Response = method(**request_args)
            if api_response.status_code not in [200, 201]: #! Added 201
                method_str = str(self.api_operation.method.value)
                response_text = (
                    f"{api_response.status_code}: {api_response.reason}"
                    + f"\nFor {method_str.upper()}  {request_args['url']}\n"
                    + f"Called with args: {request_args['params']}"
                )
            else:
                response_text = api_response.text
        except Exception as e:
            response_text = f"Error with message {str(e)}"
        response_text = response_text[: self.max_text_length]
        intermediate_steps["response_text"] = response_text
        _run_manager.on_text(
            response_text, color="blue", end="\n", verbose=self.verbose
        )
        if self.api_response_chain is not None:
            _answer = self.api_response_chain.predict_and_parse(
                response=response_text,
                instructions=instructions,
                callbacks=_run_manager.get_child(),
            )
            answer = cast(str, _answer)
            _run_manager.on_text(answer, color="yellow", end="\n", verbose=self.verbose)
            return self._get_output(answer, intermediate_steps)
        else:
            return self._get_output(response_text, intermediate_steps)