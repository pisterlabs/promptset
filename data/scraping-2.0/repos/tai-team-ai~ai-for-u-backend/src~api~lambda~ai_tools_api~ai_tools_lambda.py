import logging
import os
import sys
import traceback
from enum import Enum
from uuid import UUID
from pathlib import Path
from pydantic import BaseModel
from openai.error import RateLimitError
from mangum import Mangum
from fastapi import FastAPI, APIRouter, Request, status , Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pynamodb.models import Model

dir_path = Path(__file__).parent
sys.path.append(str(dir_path / "../dependencies"))
sys.path.append(str(dir_path / "utils"))
sys.path.append(str(dir_path / "routers"))
sys.path.append(str(dir_path))
sys.path.append(Path(__file__, "../dynamodb_models"))
from api_gateway_settings import APIGatewaySettings, DeploymentStage
from ai_tools_lambda_settings import AIToolsLambdaSettings
from dynamodb_models import UserDataTableModel
from routers import (
    text_revisor,
    cover_letter_writer,
    catchy_title_creator,
    sandbox_chatgpt,
    text_summarizer,
    feedback,
    subscription,
)
from utils import (
    prepare_response,
    UserTokenNotFoundError,
    initialize_openai,
    AUTHENTICATED_USER_ENV_VAR_NAME,
    UUID_HEADER_NAME,
    USER_TOKEN_HEADER_NAME,
    is_user_authenticated,
    EXAMPLES_ENDPOINT_POSTFIX,
    get_user_uuid_from_jwt_token,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


API_DESCRIPTION = """
This is the API for the AI for U project. It is a collection of endpoints that
use OpenAI's GPT-3 API to generate text. All requests must include a uuid header.
This uuid is used to check if the user is authenticated and to track usage of the API.
"""

lambda_settings = AIToolsLambdaSettings()
api_gateway_settings = APIGatewaySettings()
router = APIRouter()


@router.get(f"/status")
def get_status(request: Request, response: Response):
    """Return status okay."""
    response_body = {"status": "ok"}
    return response_body

def get_error_message(error: Exception) -> str:
    """Return error message."""
    traceback_str = "\n".join(traceback.format_exception(type(error), error, error.__traceback__))
    logger.error(traceback_str)
    if api_gateway_settings.deployment_stage == DeploymentStage.DEVELOPMENT.value:
        return traceback_str
    return str(error)

def get_error_response(request: Request, content: dict) -> Response:
    """Return error response."""
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content
    )
    prepare_response(response, request)
    return response

def handle_request_validation_error(request: Request, exc: RequestValidationError):
    """Handle exception."""
    msg = get_error_message(exc)
    content = {"Validation Exception": msg}
    return get_error_response(request, content)

def handle_user_token_error(request: Request, exc: UserTokenNotFoundError):
    """Handle exception."""
    msg = get_error_message(exc)
    content = {"userTokenException": msg}
    return get_error_response(request, content)

def handle_generic_exception(request: Request, exc: Exception):
    """Handle exception."""
    msg = get_error_message(exc)
    content = {"Exception Raised": msg + "\n\n" + os.environ.get("AUTHENTICATED_USER", None)}
    return get_error_response(request, content)

def handle_rate_limit_exception(request: Request, exc: RateLimitError):
    """Handle exception."""
    msg = get_error_message(exc)
    content = {"Rate Limit Exception": msg}
    return get_error_response(request, content)

def initialize_user_db(uuid: UUID, is_user_authenticated: bool):
    try:
        user_data_model: UserDataTableModel = UserDataTableModel.get(str(uuid))
    except (Model.DoesNotExist):
        user_data_model = UserDataTableModel(str(uuid), authenticated_user=is_user_authenticated)
        user_data_model.save()
    if is_user_authenticated:
        user_data_model.update(actions=[UserDataTableModel.authenticated_user.set(True)])

def create_fastapi_app():
    """Create FastAPI app."""
    root_path = f"/{api_gateway_settings.deployment_stage}"
    if api_gateway_settings.deployment_stage == DeploymentStage.LOCAL.value:
        root_path = ""

    app = FastAPI(
        root_path=root_path,
        docs_url=f"/{api_gateway_settings.openai_route_prefix}/docs",
        openapi_url=f"/{api_gateway_settings.openai_route_prefix}/openapi.json",
        redoc_url=None,
        description=API_DESCRIPTION,
    )

    @app.middleware("http")
    async def check_if_header_is_present(request: Request, call_next):
        """Check if user is authenticated."""
        path = request.url.path
        examples_paths = []
        for router_ in routers:
            for route in router_.routes:
                if route.path.endswith(EXAMPLES_ENDPOINT_POSTFIX):
                    examples_paths.append(f"{root_path}/{api_gateway_settings.openai_route_prefix}{route.path}")
        allowed_paths = {
            f"{root_path}/{api_gateway_settings.openai_route_prefix}/docs",
            f"{root_path}/{api_gateway_settings.openai_route_prefix}/openapi.json",
        }
        allowed_paths.update(examples_paths)
        uuid_str = request.headers.get(UUID_HEADER_NAME, None)
        logger.info("path: %s", allowed_paths)
        logger.info("uuid_str: %s", uuid_str)
        uuid = None
        try:
            uuid = UUID(uuid_str, version=4)
        except Exception as e: # pylint: disable=broad-except
            if path not in allowed_paths:
                raise UserTokenNotFoundError("User UUID not found.") from e
        if uuid:
            user_token = request.headers.get(USER_TOKEN_HEADER_NAME, None)
            authenticated = False
            if user_token:
                jwt_uuid = get_user_uuid_from_jwt_token(user_token)
                authenticated = is_user_authenticated(uuid, jwt_uuid)
            os.environ[AUTHENTICATED_USER_ENV_VAR_NAME] = str(authenticated)
            # uuid_to_use = jwt_uuid if authenticated else uuid # once both tokens match (future pull), we can use either, for now, we need to use the uuid as other endpoints look up user data with it
            uuid_to_use = uuid
            initialize_user_db(uuid_to_use, authenticated)
            logger.info(f"Authenticated: {authenticated}")
        response = await call_next(request)
        prepare_response(response, request)
        return response

    routers = [
        router,
        text_summarizer.router,
        text_revisor.router,
        catchy_title_creator.router,
        cover_letter_writer.router,
        sandbox_chatgpt.router,
        feedback.router,
        subscription.router,
    ] 
    initialize_openai()
    for router_ in routers:
        add_router_with_prefix(app, router_, f"/{api_gateway_settings.openai_route_prefix}")
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(Exception, handle_generic_exception)
    app.add_exception_handler(UserTokenNotFoundError, handle_user_token_error)
    app.add_exception_handler(RateLimitError, handle_rate_limit_exception)
    
    return app

def add_router_with_prefix(app: FastAPI, router: APIRouter, prefix: str) -> None:
    """Add router with prefix."""
    app.include_router(router, prefix=prefix)

def lambda_handler(event, context):
    """Lambda handler that starts a FastAPI server with uvicorn."""
    app = create_fastapi_app()
    handler = Mangum(app=app)
    return handler(event, context)
