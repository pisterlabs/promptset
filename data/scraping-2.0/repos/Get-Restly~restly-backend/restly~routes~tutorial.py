from json import dumps, loads
from typing import Optional

import openai
from flask import Blueprint, jsonify
from flask_pydantic import validate
from pydantic import BaseModel

from ..db import db
from ..models import Spec, Tutorial
from ..prompts import GENERATE_TUTORIAL_PROMPT
from ..types import ApiEndpoint, ApiEndpointList, TutorialModel, TutorialLiteModel
from ..utils import SpecFormatter
import logging

from flask import Response, stream_with_context
from openai.types.chat import ChatCompletionChunk
from openai._streaming import Stream
from .middleware import user_authenticated


tutorial_bp = Blueprint("tutorial", __name__)
logging.basicConfig(level=logging.INFO)


class ListTutorialsResponse(BaseModel):
    tutorials: list[TutorialLiteModel]


@tutorial_bp.route("/api/v1/tutorials", methods=["GET"])
@user_authenticated
@validate()
def list_tutorials(current_user):
    tutorials = Tutorial.query.filter_by(user_id=current_user.id).all()
    models = [
        TutorialLiteModel(id=tutorial.id, name=tutorial.name) for tutorial in tutorials
    ]
    return ListTutorialsResponse(tutorials=models)


class GetTutorialResponse(BaseModel):
    tutorial: TutorialModel


@tutorial_bp.route("/api/v1/tutorials/<int:id>", methods=["GET"])
@user_authenticated
@validate()
def get_tutorial(current_user, id: int):
    tutorial = Tutorial.query.filter_by(id=id, user_id=current_user.id).first()
    if not tutorial:
        return jsonify({"error": "Tutorial not found"}), 404
    model = TutorialModel(
        id=tutorial.id,
        name=tutorial.name,
        content=tutorial.content,
        query=tutorial.input,
        relevant_apis=loads(tutorial.relevant_apis),
        spec_id=tutorial.spec_id,
    )

    return GetTutorialResponse(tutorial=model)


class CreateTutorialRequest(BaseModel):
    name: str


class CreateTutorialResponse(BaseModel):
    id: int


@tutorial_bp.route("/api/v1/tutorials", methods=["POST"])
@user_authenticated
@validate()
def create_tutorial(current_user, body: CreateTutorialRequest):
    tutorial = Tutorial(name=body.name, user_id=current_user.id)
    db.session.add(tutorial)
    db.session.commit()
    return CreateTutorialResponse(id=tutorial.id)


class GenerateTutorialRequest(BaseModel):
    specId: int
    query: str
    apis: list[ApiEndpoint]
    server: str


class GenerateTutorialResponse(BaseModel):
    id: int
    content: str


@tutorial_bp.route("/api/v1/tutorials/<int:id>/generate-content", methods=["POST"])
@user_authenticated
@validate()
def generate_tutorial_content(current_user, id: int, body: GenerateTutorialRequest):
    spec: Optional[Spec] = Spec.query.filter_by(
        id=body.specId, user_id=current_user.id
    ).first()
    if not spec:
        return jsonify({"error": "Spec not found"}), 404

    tutorial = Tutorial.query.filter_by(id=id, user_id=current_user.id).first()
    if not tutorial:
        return jsonify({"error": "Tutorial not found"}), 404

    def on_completion(content_aggregated):
        tutorial.content = content_aggregated
        tutorial.input = body.query
        tutorial.relevant_apis = relevant_apis
        tutorial.spec_id = spec.id
        tutorial.server = body.server

        db.session.add(tutorial)
        db.session.commit()

    # Streams the response
    completion = generate_content(
        loads(spec.content), body.query, body.apis, body.server
    )

    def generate():
        content_aggregated = ""
        try:
            for chunk in completion:
                for choice in chunk.choices:
                    content = choice.delta.content
                    if content is not None:
                        content_aggregated += content
                        yield content
        finally:
            on_completion(content_aggregated)

    relevant_apis = ApiEndpointList(apis=body.apis).model_dump_json()

    # return GenerateTutorialResponse(
    #     id=tutorial.id,
    #     content=content,
    # )

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return Response(
        stream_with_context(generate()), headers=headers, content_type="text/plain"
    )


class UpdateTutorialContentRequest(BaseModel):
    content: str


class UpdateTutorialContentResponse(BaseModel):
    id: int


@tutorial_bp.route("/api/v1/tutorials/<int:id>/content", methods=["PUT"])
@user_authenticated
@validate()
def update_tutorial_content(current_user, id: int, body: UpdateTutorialContentRequest):
    tutorial = Tutorial.query.filter_by(id=id, user_id=current_user.id).first()
    if not tutorial:
        return jsonify({"error": "Tutorial not found"}), 404

    tutorial.content = body.content

    db.session.add(tutorial)
    db.session.commit()

    return UpdateTutorialContentResponse(id=tutorial.id)


def generate_content(
    spec: dict, query: str, apis: list[ApiEndpoint], server: str
) -> Stream[ChatCompletionChunk]:
    trimmed_spec = SpecFormatter(spec).narrow_api_list(apis)

    included_refs = SpecFormatter(spec).collect_refs(trimmed_spec)
    ref_tree = SpecFormatter(spec).select_nodes(included_refs)
    security_schemes = {
        key: value
        for key, value in SpecFormatter(spec).extract_security_info().items()
        if value
    }

    final_spec = {**trimmed_spec, **ref_tree, **security_schemes}
    final_spec_str = dumps(final_spec, indent=2)
    prompt = GENERATE_TUTORIAL_PROMPT.format(
        query=query, spec=final_spec_str, server=server
    )

    # logging.info("Prompt for genning tutorial %s", prompt)

    client = openai.OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0,
    )
    return completion
