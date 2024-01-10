import json
import os
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from openai import OpenAI

from auto_functions.database import execute_sql, execute_sql_select
from auto_functions.routes.api_specs.schemas import ApiSpec, ApiSpecCreateParams

api_spec_router = APIRouter(prefix="/api-specs")


@api_spec_router.post(
    "/",
    summary="Create an API spec",
    status_code=status.HTTP_201_CREATED,
    response_model=ApiSpec,
)
def create_api_spec(api_spec: ApiSpecCreateParams):
    api_spec_id = uuid4()
    filename = f"auto_functions/data/api_specs/{api_spec_id}.spec.json"
    with open(filename, "w") as f:
        json.dump(api_spec.spec, f, indent=4)

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    assistant_id = os.getenv("ASSISTANT_ID")
    if assistant_id is None:
        raise HTTPException(500, "ASSISTANT_ID is not set")
    with open(filename, "rb") as f:
        openai_file = openai_client.files.create(
            file=f,
            purpose="assistants",
        )
    openai_client.beta.assistants.files.create(assistant_id, file_id=openai_file.id)

    sql = (
        "INSERT INTO apispec (id, name, assistant_file_id) VALUES "
        f"('{api_spec_id}', '{api_spec.name}', '{openai_file.id}')"
    )
    execute_sql(sql)
    return ApiSpec(id=api_spec_id, name=api_spec.name, assistant_file_id=openai_file.id)


@api_spec_router.get(
    "/",
    summary="List API specs",
    response_model=list[ApiSpec],
)
def list_api_specs():
    return [ApiSpec.model_validate(dict(row)) for row in execute_sql_select("SELECT * FROM apispec")]
