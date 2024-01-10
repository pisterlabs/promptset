from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from logzero import logger

from local_utils.settings import StreamlitAppSettings


def get_completion(prompt: str) -> str:
    # Your PAT (Personal Access Token) can be found in the portal under Authentification
    settings = StreamlitAppSettings.load()
    # Specify the correct user_id/app_id pairings
    # Since you're making inferences outside your app's scope
    USER_ID = "openai"
    APP_ID = "chat-completion"
    # Change these to whatever model and text URL you want to use
    MODEL_ID = "GPT-4"
    # MODEL_VERSION_ID = "ad16eda6ac054796bf9f348ab6733c72"

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (("authorization", "Key " + settings.clarifai_pat.get_secret_value()),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    logger.info("Getting chat completion with GPT-4")
    logger.debug("PROMPT")
    logger.debug(prompt)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,
            model_id=MODEL_ID,
            # version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=prompt)))],
        ),
        metadata=metadata,
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]
    logger.debug("RESPONSE")
    logger.debug(output.data.text.raw)
    return output.data.text.raw


def get_completion_openai(prompt: str) -> str:
    from langchain.llms import OpenAIChat

    logger.info("Getting chat completion with GPT-4")
    logger.debug("PROMPT")
    logger.debug(prompt)

    chat = OpenAIChat(model_name="gpt-4")
    response = chat(prompt)
    logger.debug("RESPONSE")
    logger.debug(response)
    return response
