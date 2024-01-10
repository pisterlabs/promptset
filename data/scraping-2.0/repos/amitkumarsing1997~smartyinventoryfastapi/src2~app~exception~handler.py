

from typing import Any
from fastapi import Request,FastAPI,HTTPException
from fastapi.responses import JSONResponse
from src2.app.exception.custom_exception import GenericException
from src2.app.shared.response import Response
# from starlette.exceptions import HTTPException as StarletteHTTPException

def exception_handlers(app : FastAPI):
    @app.exception_handler(GenericException)
    def generic_exception_handler(request:Request , exc:GenericException):
        return JSONResponse(
            status_code=200,
            content=Response[Any](success=False, msg=exc.msg, msg_code=exc.msg_code, body=exc.body).model_dump()
        )

    @app.exception_handler(HTTPException)
    def http_exception_handler(request:Request, exc:HTTPException):
        return JSONResponse(
            status_code=200,
            content=Response[Any](success=False, msg=exc.detail, msg_code=str(exc.status_code),body=exc.headers).model_dump()
        )


    @app.exception_handler(Exception)
    def other_exceptions(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=Response[Any](success=False, msg='Exception occurred while procession your request',
                                  msg_code='server_error', body=None).model_dump()
        )















# from fastapi import FastAPI, HTTPException
# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import PlainTextResponse
# from starlette.exceptions import HTTPException as StarletteHTTPException
#
# app = FastAPI()
#
#
# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request, exc):
#     return PlainTextResponse(str(exc.detail), status_code=exc.status_code)
#
#
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request, exc):
#     return PlainTextResponse(str(exc), status_code=400)
#
#
# @app.get("/items/{item_id}")
# async def read_item(item_id: int):
#     if item_id == 3:
#         raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
#     return {"item_id": item_id}
#
#
#
#
#
#
#
#
#
#
#
#
# from typing import Any
#
# import openai
# from fastapi import Request, FastAPI
# from fastapi.responses import JSONResponse
#
# from src.app.exception.custom_exception import GenericException
# from src.app.shared.app_const import APIMsgCode
# from src.app.shared.response import Response
#
#
# def exception_handlers(app: FastAPI):
#
#     @app.exception_handler(GenericException)
#     def generic_exception_handler(request: Request, exc: GenericException):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, msg=exc.msg, msg_code=exc.msg_code, body=exc.body).model_dump()
#         )
#
#     # open ai exception handling
#     @app.exception_handler(openai.BadRequestError)
#     def openai_bad_request_exception(request: Request, exc: openai.BadRequestError):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, body=exc.message, msg_code=APIMsgCode.GPT_REQ_INV_ERR, msg="Please send the valid data").model_dump()
#         )
#
#     @app.exception_handler(openai.RateLimitError)
#     def openai_bad_request_exception(request: Request, exc: openai.RateLimitError):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, body=exc.message, msg_code=APIMsgCode.GPT_REQ_LIMIT_CROSS_ERR, msg="You have crossed your request limit.").model_dump()
#         )
#
#     @app.exception_handler(openai.APITimeoutError)
#     def openai_bad_request_exception(request: Request, exc: openai.APITimeoutError):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, body=exc.message, msg_code=APIMsgCode.GPT_REQ_TIMEOUT_ERR, msg="Request timeout, Please try again").model_dump()
#         )
#
#     @app.exception_handler(openai.APIError)
#     def openai_bad_request_exception(request: Request, exc: openai.APIError):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, body=exc.message, msg_code=APIMsgCode.GPT_API_ERR, msg="Some error occurred while processing your request").model_dump()
#         )
#
#     @app.exception_handler(openai.BadRequestError)
#     def openai_bad_request_exception(request: Request, exc: openai.BadRequestError):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, body=exc.message, msg_code=APIMsgCode.GPT_REQ_INV_ERR).model_dump()
#         )
#
#     @app.exception_handler(openai.BadRequestError)
#     def openai_bad_request_exception(request: Request, exc: openai.BadRequestError):
#         return JSONResponse(
#             status_code=200,
#             content=Response[Any](success=False, body=exc.message, msg_code=APIMsgCode.GPT_REQ_INV_ERR).model_dump()
#         )
#
#     @app.exception_handler(Exception)
#     def other_exceptions(request: Request, exc: Exception):
#         return JSONResponse(
#             status_code=500,
#             content=Response[Any](success=False, msg='Exception occurred while procession your request',
#                                   msg_code='server_error', body=None).model_dump()
#         )
