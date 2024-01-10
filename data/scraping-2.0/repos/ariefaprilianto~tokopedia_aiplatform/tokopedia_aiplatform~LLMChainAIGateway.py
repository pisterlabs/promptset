from typing import Any, Callable, Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import LLMResult
from langchain import LLMChain
from google.cloud.aiplatform_v1.services.prediction_service import transports
from google.cloud.aiplatform_v1.types import prediction_service

class LLMChainAIGateway(LLMChain):

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""

        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)

        for input_dict in input_list:
            request_id = input_dict.get('request_id')
            if request_id:
                self.llm_kwargs['request_id'] = request_id

        return self.llm.generate_prompt(
            prompts,
            stop,
            callbacks=run_manager.get_child() if run_manager else None,
            **self.llm_kwargs,
        )

class PredictionServiceGrpcTransportAIGateway(transports.PredictionServiceGrpcTransport):
    @property
    def predict(
        self,
    ) -> Callable[
        [prediction_service.PredictRequest], prediction_service.PredictResponse
    ]:
        r"""Return a callable for the predict method over gRPC.

        Perform an online prediction.

        Returns:
            Callable[[~.PredictRequest],
                    ~.PredictResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if "predict" not in self._stubs:
            self._stubs["predict"] = self.grpc_channel.unary_unary(
                "/tokopedia.cloud.aiplatform.v1.PredictionService/Predict",
                request_serializer=prediction_service.PredictRequest.serialize,
                response_deserializer=prediction_service.PredictResponse.deserialize,
            )
        return self._stubs["predict"]