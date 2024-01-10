from fastapi import status
from typing import Any, Dict


class ReqResponses:
    # class Responses(Dict[str, Any]):
    """Responses for fastapi endpoints responses. Used to pass fastapi methods for OpenAPI documentation.
            Usage example: `responses={**Responses.INVALID_ITEM_ID, **Responses.PATH_UPDATED}`
    )
    """

    rtype = Dict[str | int, Any]  # mypy type error otherwise

    INVALID_ITEM_ID: rtype = {
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Invalid itemid supplied"
        },
        status.HTTP_404_NOT_FOUND: {"description": "Item not found"},
    }

    NO_ITEM_UPDATED: rtype = {
        status.HTTP_400_BAD_REQUEST: {"description": "No item updated"}
    }

    POST_CREATED: rtype = {status.HTTP_201_CREATED: {"description": "Item created"}}

    PATH_UPDATED: rtype = {status.HTTP_200_OK: {"description": "Item updated"}}

    NOT_IMPLEMENTED: rtype = {
        status.HTTP_405_METHOD_NOT_ALLOWED: {"description": "Method not implemented"}
    }

    OPENAI_ERROR: rtype = {
        status.HTTP_502_BAD_GATEWAY: {"description": "Error from OpenAI API Call"}
    }

    OPENAI_TIMEOUT: rtype = {
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "OpenAI API Call timeout"}
    }

    MALFORMED_REQUEST: rtype = {
        status.HTTP_400_BAD_REQUEST: {"description": "Malformed request"}
    }

    NOT_AUTHENTICATED: rtype = {
        status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"}
    }

    INVALID_USERNAME_OR_PASSWORD: rtype = {
        status.HTTP_401_UNAUTHORIZED: {"description": "Incorrect username or password"}
    }

    EMAIL_ALREADY_EXISTS: rtype = {
        status.HTTP_409_CONFLICT: {"description": "Email already exists"}
    }

    PATCH_RESPONSES: rtype = {
        **INVALID_ITEM_ID,
        **PATH_UPDATED,
        **NO_ITEM_UPDATED,
        **NOT_AUTHENTICATED,
        **MALFORMED_REQUEST,
    }

    GET_RESPONSES: rtype = {
        **INVALID_ITEM_ID,
        **MALFORMED_REQUEST,
        **NOT_AUTHENTICATED,
    }

    POST_RESPONSES: rtype = {
        **MALFORMED_REQUEST,
        **NOT_AUTHENTICATED,
        **POST_CREATED,
    }
