from typing import List, Annotated
from fastapi import APIRouter, Depends, Request, status

from src.core.database import get_db
from src.crud.org import org_crud
from src.schemas.user import UserInDB
from src.core.user import get_current_active_user
from src.core.completition import get_openai_chat_completition

from src.core.utils import str_to_ObjectId
from src.crud.inference import inference_crud
from src.endpoints.ApiResponses import ReqResponses
from src.endpoints.endpoint_exceptions import (
    ItemNotFoundException,
    MalformedRequestError,
    OpenAIError,
)

from src.schemas.inference import (
    InferenceCreateByPromptId,
    InferenceCreateByPromptVersionId,
    InferencePostResponse,
    InferenceRead,
    InferenceResponseError,
    InferenceResponseStatus,
    InferenceUpdate,
)


router = APIRouter()

# TODO: feedbacks. distiguish end user and 3rd party feedback. should be in separate collection


# TODO: add filter by status
# TODO: add pagination or cursor
# TODO: ordering?
@router.get(
    "/inferences",
    tags=["inferences"],
    response_model=List[InferenceRead],
    responses={**ReqResponses.GET_RESPONSES},
)
async def get_inferences_list(
    current_user: Annotated[UserInDB, Depends(get_current_active_user)],
    db=Depends(get_db),
    prompt_version_id: str | None = None,
    prompt_id: str | None = None,
):
    if prompt_version_id and prompt_id:
        raise MalformedRequestError(
            "Can't filter by both `prompt_version_id` and `prompt_id`. Choose one."
        )

    filter = None

    if prompt_version_id:
        prompt_version_id_oid = str_to_ObjectId(
            prompt_version_id
        )  # raises if malformed
        filter = {"prompt_version_id": prompt_version_id_oid}
    elif prompt_id:
        prompt_id_oid = str_to_ObjectId(prompt_id)  # raises if malformed
        filter = {"prompt_id": prompt_id_oid}

    inferences = await inference_crud.get_multi(db, current_user, filter=filter)

    return inferences


@router.get(
    "/inferences/{id}",
    tags=["inferences"],
    response_model=InferenceRead,
    responses={**ReqResponses.GET_RESPONSES},
)
async def get_inference_by_id(
    id: str,
    current_user: Annotated[UserInDB, Depends(get_current_active_user)],
    db=Depends(get_db),
):
    return await inference_crud.get(db, id, current_user)


# TODO: Inference by prompt_id - should use one of (or the latest?) live version
# TODO: ability to log inference request and response separately and without calling completition - so one can call their completion on their own or log historic data
@router.post(
    "/inferences",
    tags=["inferences"],
    status_code=status.HTTP_201_CREATED,
    responses={
        **ReqResponses.POST_RESPONSES,
        **ReqResponses.OPENAI_ERROR,
        **ReqResponses.OPENAI_TIMEOUT,
    },
)
async def new_inference(
    request: Request,
    inferenceRequest: InferenceCreateByPromptVersionId | InferenceCreateByPromptId,
    current_user: Annotated[UserInDB, Depends(get_current_active_user)],
    db=Depends(get_db),
) -> InferencePostResponse:
    """The core of the functionality:
    1. Populating the template from prompt version with the passed values
    2. Logging the request
    3. Sending request to provider
    4. Logging response
    4. Returning response with `inference_id`

    You can specify which prompt version you want to use in two ways by setting on of:

     - `prompt_version_id` - uses the specified prompt version
     - `prompt_id` - uses the `Live` status prompt version assigned to the given `prompt_id`. This allows to release new prompt versions using Prompton API if you only reference `prompt_id` in your client code.

         If there are multiple  prompt versions in `Live` status for the prompt_id then it picks one randomly. It's useful for split testing.
         This method will return an error if there is no `Live` status message.

    It also handles errors, timeouts and sets the inference status accordingly. It will still process response if client disconnects before it arrives.

    _Note: raw request data is also logged, GET `inferences/{id}` reponse includes it as well._

    You can use a few easter eggs to test it without a valid api key:

      - `"end_user_id": "mock_me_softly"`
      - `"end_user_id": "timeout_me_softly"`
      - `"end_user_id": "fail_me_softly"`

    """

    # TODO: refactor response schema - it's a pain to get id in client code from OPENAI_ERROR | OPENAI_TIMEOUT

    try:
        user_org = await org_crud.get(db, current_user.org_id, current_user)
    except ItemNotFoundException:
        raise MalformedRequestError(
            f"Current user is not associated with a valid organization - user's org_id {current_user.org_id} doesn't exists. Can't create inferences."
        )

    if not user_org.access_keys or not user_org.access_keys.get("openai_api_key"):
        raise MalformedRequestError(
            f"Curren user's organization doesn't have openai_api_key set. Can't create inferences. Set access_keys.openai_api_key for org_id {user_org.id}."
        )

    openai_api_key = user_org.access_keys.get("openai_api_key")

    if not openai_api_key or not openai_api_key.strip():
        raise MalformedRequestError(
            f"Curren user's organization openai_api_key key is empty. Can't create inferences. Set access_keys.openai_api_key for org_id {user_org.id}."
        )

    new_inference = await inference_crud.create(db, inferenceRequest, current_user)

    # we could avoid re-fetching by refactoring inference_crud.create - get back to it if performanance is an issue
    inferenceDB = await inference_crud.get(db, new_inference.inserted_id, current_user)

    inferenceResponse = await get_openai_chat_completition(
        inferenceDB.request.raw_request,
        openai_api_key,
        request_timeout=inferenceRequest.request_timeout,
    )

    #  this logic should be in inference_crud.update but requires refactor to get back calculated status without refetching
    if isinstance(inferenceResponse, InferenceResponseError):
        inferenceResponse.isError = True

        if inferenceResponse.error.error_class == "openai.error.Timeout":
            status = InferenceResponseStatus.COMPLETITION_TIMEOUT
        else:
            status = InferenceResponseStatus.COMPLETITION_ERROR

    else:
        inferenceResponse.isError = False
        status = InferenceResponseStatus.PROCESSED

    inferenceResponse.is_client_connected_at_finish = (
        not await request.is_disconnected()
    )

    inferenceUpdate = InferenceUpdate(status=status, response=inferenceResponse)
    _ = await inference_crud.update(db, inferenceDB.id, inferenceUpdate, current_user)

    if isinstance(inferenceResponse, InferenceResponseError):
        if status == InferenceResponseStatus.COMPLETITION_TIMEOUT:
            raise OpenAIError(
                inference_id=inferenceDB.id,
                inferenceResponse=inferenceResponse,
                message="OpenAI API Timeout Error",
            )
        else:  # status == InferenceResponseStatus.COMPLETITION_ERROR
            raise OpenAIError(
                inference_id=inferenceDB.id,
                inferenceResponse=inferenceResponse,
                message="OpenAI API Error",
            )

    response = InferencePostResponse(id=inferenceDB.id, response=inferenceResponse)

    return response
