import asyncio
import json

from quart import Blueprint, request, jsonify, websocket
from quart_cors import route_cors
from beddybai.generation.api import OpenAI, user

from beddybai.generation.generator import StoryGenerator

from ..database import db
from ..database.models import Artist, Author, Story, Theme, Title, Lesson

stories_blueprint = Blueprint("stories", __name__)
themes = []


@stories_blueprint.route("/start", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def start_new_story():
    data = await request.get_json()
    data.pop("api_key", None)

    if not all(((age_min := data.get("age_min")), (age_max := data.get("age_max")))):
        return (
            {
                "error": "Missing required parameter(s)",
                "required_parameters": ["age_min", "age_max"],
            },
            400,
        )

    story = Story(age_min=age_min, age_max=age_max)

    db.session.add(story)
    db.session.commit()

    return jsonify({"message": "Story started", "story": story.id}), 201


@stories_blueprint.route("/themes", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def generate_theme_suggestions():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = await request.get_json()
    data.pop("api_key", None)

    return await story_generator.generate_story_themes(**data)


@stories_blueprint.route("/lessons", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def generate_lesson_suggestions():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = await request.get_json()
    data.pop("api_key", None)

    return await story_generator.generate_story_lessons(**(data or {}))


@stories_blueprint.route("/titles", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def generate_title_suggestions():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = await request.get_json()
    data.pop("api_key", None)

    return await story_generator.generate_story_titles(**(data or {}))


@stories_blueprint.route("/authors", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def generate_author_suggestions():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = await request.get_json()
    data.pop("api_key", None)

    return await story_generator.generate_author_styles(**(data or {}))


@stories_blueprint.route("/artists", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def generate_artist_suggestions():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = await request.get_json()
    data.pop("api_key", None)

    return await story_generator.generate_artist_styles(**(data or {}))


@stories_blueprint.route("/<int:story_id>/theme", methods=["PATCH"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def update_story_theme(story_id: int):
    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    if not all(
        ((emoji := data.get("emoji")), (story_theme := data.get("story_theme")))
    ):
        return (
            {
                "error": "Missing required parameter(s)",
                "required_parameters": ["emoji", "story_theme"],
            },
            400,
        )

    theme = Theme.query.filter_by(emoji=emoji, story_theme=story_theme).first()

    if not theme:
        theme = Theme(**data)

        db.session.add(theme)
        db.session.commit()

    story = Story.query.get_or_404(story_id)
    story.theme_id = theme.id

    db.session.commit()
    return story.to_json()


@stories_blueprint.route("/<int:story_id>/title", methods=["PATCH"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def update_story_title(story_id: int):
    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    if not (story_title := data.get("story_title")):
        return (
            {
                "error": "Missing required parameter(s)",
                "required_parameters": ["story_title"],
            },
            400,
        )

    title = Title.query.filter_by(story_title=story_title).first()

    if not title:
        title = Title(**data)

        db.session.add(title)
        db.session.commit()

    story = Story.query.get_or_404(story_id)
    story.title_id = title.id

    db.session.commit()
    return story.to_json()


@stories_blueprint.route("/<int:story_id>/lesson", methods=["PATCH"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def update_story_lesson(story_id: int):
    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    if not (story_lesson := data.get("story_lesson")):
        return (
            {
                "error": "Missing required parameter(s)",
                "required_parameters": ["story_lesson"],
            },
            400,
        )

    lesson = Lesson.query.filter_by(story_lesson=story_lesson).first()

    if not lesson:
        lesson = Lesson(**data)

        db.session.add(lesson)
        db.session.commit()

    story = Story.query.get_or_404(story_id)
    story.lesson_id = lesson.id

    db.session.commit()
    return story.to_json()


@stories_blueprint.route("/<int:story_id>/author", methods=["PATCH"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def update_story_author(story_id: int):
    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    if not all(
        (
            (author_name := data.get("author_name")),
            (author_style := data.get("author_style")),
        )
    ):
        return (
            {
                "error": "Missing required parameter(s)",
                "required_parameters": ["author_name", "author_style"],
            },
            400,
        )

    author = Author.query.filter_by(
        author_name=author_name, author_style=author_style
    ).first()

    if not author:
        author = Author(**data)

        db.session.add(author)
        db.session.commit()

    story = Story.query.get_or_404(story_id)
    story.author_id = author.id

    db.session.commit()
    return story.to_json()


@stories_blueprint.route("/<int:story_id>/artist", methods=["PATCH"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def update_story_artist(story_id: int):
    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    if not all(
        (
            (artist_name := data.get("artist_name")),
            (artist_style := data.get("artist_style")),
        )
    ):
        return (
            {
                "error": "Missing required parameter(s)",
                "required_parameters": ["artist_name", "artist_style"],
            },
            400,
        )

    artist = Artist.query.filter_by(
        artist_name=artist_name, artist_style=artist_style
    ).first()

    if not artist:
        artist = Artist(**data)

        db.session.add(artist)
        db.session.commit()

    story = Story.query.get_or_404(story_id)
    story.artist_id = artist.id

    db.session.commit()
    return story.to_json()


@stories_blueprint.websocket("/themes/stream")
async def stream_themes():
    story_generator = None

    async def parse_and_emit_objects(**kwargs):
        async for theme in story_generator.generate_story_themes_streaming(**kwargs):
            themes.append(theme)
            response = {"type": "item", "data": theme}
            print(f"->->-> {response}")

            await websocket.send(json.dumps(response))

        end = {"type": "end"}
        print(f"->->-> {end}")
        await websocket.send(json.dumps(end))

    while True:
        message = await websocket.receive()
        data = json.loads(message)

        if data.get("type") == "request":
            req = data.get("data")
            story_generator = StoryGenerator(req.pop("api_key"))
            asyncio.create_task(parse_and_emit_objects(**req))


@stories_blueprint.websocket("/lessons/stream")
async def stream_lessons():
    story_generator = None

    async def parse_and_emit_objects(**kwargs):
        async for lesson in story_generator.generate_story_lessons_streaming(**kwargs):
            await websocket.send(
                json.dumps({"type": "item", "data": {"story_lesson": lesson}})
            )

            await websocket.send(json.dumps(response))

        end = {"type": "end"}
        print(f"->->-> {end}")
        await websocket.send(json.dumps(end))

    while True:
        message = await websocket.receive()
        data = json.loads(message)
        print(f"<-- {data}")

        if data.get("type") == "request":
            req = data.get("data")
            story_generator = StoryGenerator(req.pop("api_key", ""))
            asyncio.create_task(parse_and_emit_objects(**req))


@stories_blueprint.websocket("/lessons/stream")
async def stream_lessons():
    story_generator = None

    async def parse_and_emit_objects(**kwargs):
        async for lesson in story_generator.generate_story_lessons_streaming(**kwargs):
            await websocket.send(
                json.dumps({"type": "item", "data": {"story_lesson": lesson}})
            )

        await websocket.send(json.dumps({"type": "end"}))

    while True:
        message = await websocket.receive()
        data = json.loads(message)

        if data.get("type") == "request":
            req = data.get("data")
            story_generator = StoryGenerator(req.pop("api_key", ""))
            asyncio.create_task(parse_and_emit_objects(**req))


@stories_blueprint.route("/next", methods=["POST"])
# @route_cors(allow_origin=allowed_origins, allow_headers=allowed_headers)
async def get_page():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    previous_paragraphs = data.pop("previous_paragraphs", [])
    return await story_generator.generate_story_paragraph(
        info=data, previous_paragraphs=previous_paragraphs
    )


@stories_blueprint.websocket("/stream")
async def stream_pages():
    story_generator = None

    async def parse_and_emit_objects(data):
        async for paragraph in story_generator.generate_story_paragraphs_streaming(
            data
        ):
            await websocket.send(json.dumps({"type": "item", "data": paragraph}))

        await websocket.send(json.dumps({"type": "end"}))

    while True:
        message = await websocket.receive()
        data = json.loads(message)

        if data.get("type") == "request":
            req = data.get("data")
            story_generator = StoryGenerator(req.pop("api_key", ""))
            asyncio.create_task(parse_and_emit_objects(req))


@stories_blueprint.route("/image", methods=["POST"])
# @route_cors(allow_origin=allowed_origins)
async def get_image():
    openai_api_key = request.headers.get("Authorization").split(" ")[1]
    story_generator = StoryGenerator(openai_api_key)

    data = (await request.get_json()) or {}
    data.pop("api_key", None)

    story_paragraph = data.pop("story_paragraph", "")
    return await story_generator.generate_image(
        story_paragraph=story_paragraph, story_info=data, size=512
    )
