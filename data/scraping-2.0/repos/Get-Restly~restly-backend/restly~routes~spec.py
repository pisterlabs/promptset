from json import dumps, loads
from typing import Optional

import openai
import requests
from flask import Blueprint, Flask, jsonify
from flask_pydantic import validate
from pydantic import BaseModel

from ..db import db
from ..models import Spec
from ..prompts import RELEVANT_APIS_PROMPT
from ..types import ApiEndpoint, SpecModel, SpecLiteModel
from ..utils import SpecFormatter
from .middleware import user_authenticated


spec_bp = Blueprint("spec", __name__)


DEFAULT_SPEC_TITLE = "unknown"
DEFAULT_SPEC_VERSION = "1.0.0"


class ListSpecsResponse(BaseModel):
    specs: list[SpecModel]


@spec_bp.route("/api/v1/specs", methods=["GET"])
@user_authenticated
@validate()
def list_specs(current_user):
    specs = Spec.query.filter_by(user_id=current_user.id).all()
    models = [
        SpecModel(id=spec.id, name=spec.name, url=spec.url, content=spec.content)
        for spec in specs
    ]
    return ListSpecsResponse(specs=models)


class GetSpecResponse(BaseModel):
    spec: SpecModel


@spec_bp.route("/api/v1/specs/<int:id>", methods=["GET"])
@user_authenticated
@validate()
def get_spec(current_user, id: int):
    spec: Optional[Spec] = Spec.query.filter_by(id=id, user_id=current_user.id).first()
    if not spec:
        jsonify({"error": "Spec not found"}), 404
    model = SpecModel(id=spec.id, name=spec.name, url=spec.url, content=spec.content)
    return GetSpecResponse(spec=model)


class CreateSpecRequest(BaseModel):
    url: str


class CreateSpecResponse(BaseModel):
    id: int
    name: str
    spec: dict  # TODO: define spec model https://swagger.io/specification/


@spec_bp.route("/api/v1/specs", methods=["POST"])
@user_authenticated
@validate()
def create_spec(current_user, body: CreateSpecRequest):
    content = requests.get(body.url).json()

    spec_info = content.get("info", {})
    spec_title = spec_info.get("title", DEFAULT_SPEC_TITLE)
    spec_version = spec_info.get("version", DEFAULT_SPEC_VERSION)
    name = f"{spec_title} - {spec_version}"

    spec = Spec(
        name=name,
        url=body.url,
        content=dumps(content),
        user_id=current_user.id,
    )

    db.session.add(spec)
    db.session.commit()

    return CreateSpecResponse(
        id=spec.id,
        name=spec.name,
        spec=content,
    )


class RelevantApisRequest(BaseModel):
    query: str
    count: int = 10


class RelevantApisResponse(BaseModel):
    apis: list[ApiEndpoint]


@spec_bp.route("/api/v1/specs/<int:id>/relevant-apis", methods=["POST"])
@user_authenticated
@validate()
def relevant_apis(current_user, id: int, body: RelevantApisRequest):
    spec = Spec.query.filter_by(id=id, user_id=current_user.id).first()
    if not spec:
        jsonify({"error": "Spec not found"}), 404

    spec_content = loads(spec.content)
    trimmed_spec = SpecFormatter(spec_content).trim_paths_only()
    trimmed_spec_str = dumps(trimmed_spec)

    prompt = RELEVANT_APIS_PROMPT.format(query=body.query, spec=trimmed_spec_str, count=body.count)
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    resp = loads(completion.choices[0].message.content)
    return RelevantApisResponse(**resp)
