from flask import Blueprint, request, g, abort, Response
import psycopg
from .auth import authenticate
from .location import Point, get_place_info
from time import sleep
import openai
import openai.embeddings_utils as emb
from datetime import datetime, timedelta

bp = Blueprint("event", __name__, url_prefix="/event")

@bp.before_app_request
def setup_openai():
    openai.api_key_path = request.environ["OPENAI_KEY_PATH"]

def to_datetime(json: dict, end=False):
    return datetime(int(json["year"]), int(json["month"]), int(json["day"]),
                    int(json.get("hour", 23 if end else 0)), int((59 if end else 0) if "hour" not in json else json.get("minute", 0)))

def from_datetime(dt: datetime):
    return {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute
    }

def validate_create_inputs(displayname: str, start: dict, end: dict, location: str, description: str | None = None, **kwargs):
    """
    Ensures the given event information is valid.

    Returns a list of errors (see docstring for `create()`), or an empty list if no issues were found.
    """
    errors = []
    if not displayname.strip():
        errors.append({"field": "displayname", "description": "You must enter a display name."})
    if len(displayname) > 255:
        errors.append({"field": "displayname", "description": "Display name must be less than 256 characters long."})
    if description is not None and len(description) > 10000:
        errors.append({"field": "description", "description": "Description must be at most 10000 characters long."})
    startdt = enddt = None
    try:
        startdt = to_datetime(start)
    except ValueError:
        errors.append({"field": "start", "description": "Not a valid date."})
    try:
        startdt = to_datetime(end, end=True)
    except ValueError:
        errors.append({"field": "end", "description": "Not a valid date."})
    if startdt is not None and enddt is not None and startdt > enddt:
        errors.append({"field": "end", "description": "End date cannot be earlier than start date."})
    if not errors:
        if get_place_info(location) is None:
            errors.append({"field": "location", "description": "Must be a valid Place ID."})
    return errors

def create_event(displayname: str, start: dict, end: dict, location: str, description: str | None = None, **kwargs) -> int:
    """
    Adds an event with the given information to the database.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    emb_str = displayname.strip().replace("\n", " ")
    if description is not None:
        emb_str += " " + description.strip().replace("\n", " ")
    placename, _, _ = get_place_info(location)
    if placename is not None:
        emb_str += " " + placename.strip().replace("\n", " ")
    embedding = emb.get_embedding(emb_str, engine="text-embedding-ada-002", user=str(g.userid))
    with g.conn.cursor() as cur:
        cur.execute("INSERT INTO events(displayname, start, \"end\", place, host, embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;", 
                    (displayname, to_datetime(start), to_datetime(end, end=True), location, g.userid, embedding))
        id, = cur.fetchone()
        if description is not None:
            cur.execute("UPDATE events SET description = %s WHERE id = %s;", (description, id))
        return id

@bp.route("", methods=["POST"])
@authenticate
def create() -> Response:
    """
    API endpoint for creating a new event.

    Requires a JSON object in the request body with the following properties:
        - `displayname`: the name of the event
        - `description` (optional): a description of the event
        - `start`, `end`: the start and end datetimes of the event, as objects with the following integer attributes:
            - `year`, `month`, `day`
            - `hour`, `minute`
        - `location`: the Google Place ID of the event's venue
    
    The input data must meet the following requirements:
        - `displayname` must not be empty.
        - `displayname` must contain no more than 255 characters.
        - `description` must contain no more than 10000 characters.
        - `start` and `end` must be valid datetimes.
        - `end` cannot come before `start`.
        - `location` must be a valid Place ID.
    
    The following status codes will be returned:
        - 401: Need to authenticate.
        - 415: Request body is not in JSON format.
        - 400: JSON syntax is invalid, or object does not contain the right fields.
        - 422: One or more of the event options are invalid.
        - 200: Changes were successful.
    
    If a 422 status code is returned, the response body will contain a JSON array of errors in the following format:
        - `field`: the field in which the error occurred
        - `description`: a description of the error
    
    If a 200 status code is returned, the response body will contain a JSON object containing the following properties:
        - `id`: the ID of the event
    """
    if request.json is None:
        abort(415)
    if not isinstance(request.json, dict):
        abort(400)
    if "displayname" not in request.json or not isinstance(request.json["displayname"], str):
        abort(400)
    if "description" in request.json and request.json["description"] is not None \
        and not isinstance(request.json["description"], str):
        abort(400)
    if "location" not in request.json or not isinstance(request.json["location"], str):
        abort(400)
    if "start" not in request.json or not isinstance(request.json["start"], dict):
        abort(400)
    if "year" not in request.json["start"] or "month" not in request.json["start"] or "day" not in request.json["start"] \
        or ("minute" in request.json["start"] and "hour" not in request.json["start"]):
        abort(400)
    if "end" not in request.json or not isinstance(request.json["end"], dict):
        abort(400)
    if "year" not in request.json["end"] or "month" not in request.json["end"] or "day" not in request.json["end"] \
        or ("minute" in request.json["end"] and "hour" not in request.json["end"]):
        abort(400)
    try:
        _ = int(request.json["start"]["year"])
        _ = int(request.json["start"]["month"])
        _ = int(request.json["start"]["day"])
        _ = int(request.json["start"].get("hour", 0))
        _ = int(request.json["start"].get("minute", 0))
        _ = int(request.json["end"]["year"])
        _ = int(request.json["end"]["month"])
        _ = int(request.json["end"]["day"])
        _ = int(request.json["end"].get("hour", 0))
        _ = int(request.json["end"].get("minute", 0))
    except ValueError:
        abort(400)
    errors = validate_create_inputs(**request.json)
    if errors:
        return (errors, 422)
    return {"id": create_event(**request.json)}


def list_events():
    """
    Return a list of events that the logged-in user is attending and/or hosting.

    Requires `g.userid` to be set (i.e., a function that calls this must be wrapped in `@authenticate`).

    Returns a JSON object as described in the docstring for `event_list()`.
    """
    attending = []
    hosting = []

    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        cur.execute("SELECT events.id, events.displayname, events.start, events.\"end\", events.description, places.name, places.address, places.coords FROM ((attendees JOIN events ON attendees.eventid = events.id) JOIN places ON events.place = places.id) WHERE attendees.userid = %s;",
                    (g.userid,))
        for row in cur:
            id, dname, start, end, description, pname, addr, coords = row
            assert isinstance(coords, Point)
            attending.append({"id": id, "displayname": dname, "start": from_datetime(start), "end": from_datetime(end), "description": description, "venue": pname, "address": addr, "coords": {"lat": coords.lat, "lon": coords.lon}})
        cur.execute("SELECT events.id, events.displayname, events.start, events.\"end\", events.description, places.name, places.address, places.coords FROM (events JOIN places ON events.place = places.id) WHERE host = %s;", (g.userid,))
        for row in cur:
            id, dname, start, end, description, pname, addr, coords = row
            assert isinstance(coords, Point)
            hosting.append({"id": id, "displayname": dname, "start": from_datetime(start), "end": from_datetime(end), "description": description, "venue": pname, "address": addr, "coords": {"lat": coords.lat, "lon": coords.lon}})
    return {"attending": attending, "hosting": hosting}

@bp.route("", methods=["GET"])
@authenticate
def event_list():
    """
    Get a list of events that the logged-in user is attending and that they are hosting.

    Requires HTTP Basic Authentication.

    Input: None

    Status Codes:
        - 401: Need to authenticate.
        - 200: Request successful.
    
    Output: a JSON object with the following properties:
        - `attending`: a list of events that the user is attending, with each event in the following form:
            - `id`: the database ID of the event
            - `displayname`: the display name of the event
            - `start`, `end`: the start and end datetimes of the event, as objects with the following integer attributes:
                - `year`, `month`, `day`
                - `hour`, `minute`
            - `description`: the description of the event
            - `venue`: the name of the event's venue, or null
            - `address`: the address of the event's venue
            - `coords`: the coordinates of the event's venue, with the following properties:
                - `lat`: latitude
                - `lon`: longitude
        - `hosting`: a list of events that the user is hosting, with events in the same form as those in `attending`.
    """
    return list_events()


def get_event_info(id: int) -> dict:
    """
    Gets the info of the event with the given ID.

    Requires g.userid to be set (i.e., a function that calls it should be wrapped with @authenticate).
    
    Returns a JSON object with the event data (see docstring for `event_get()`), or None if it does not exist.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    info = None
    with g.conn.cursor() as cur:
        cur.execute("SELECT events.displayname, events.start, events.\"end\", events.description, events.host, places.name, places.address, places.coords FROM (events JOIN places ON events.place = places.id) WHERE events.id = %s;", (id,))
        if cur.rowcount == 0:
            return None
        name, start, end, description, host, pname, addr, coords = cur.fetchone()
        assert isinstance(coords, Point)
        info = {"displayname": name, "start": from_datetime(start), "end": from_datetime(end), "description": description, "host": {"id": host}, "venue": pname, "address": addr, "coords": {"lat": coords.lat, "lon": coords.lon}, "attendees": [], "attending": False}
        cur.execute("SELECT username, displayname FROM users WHERE id = %s;", (host,))
        uname, udname = cur.fetchone()
        info["host"]["username"] = uname
        info["host"]["displayname"] = udname
        cur.execute("SELECT attendees.userid, users.username, users.displayname FROM (attendees JOIN users ON users.id = attendees.userid) WHERE eventid = %s;", (id,))
        for row in cur:
            uid, uname, udname = row
            info["attendees"].append({"id": uid, "username": uname, "displayname": udname})
            if uid == g.userid:
                info["attending"] = True
    return info

@bp.route("/<int:eventid>", methods=["GET"])
@authenticate
def event_get(eventid: int) -> Response:
    """
    Gets the information of the event associated with the given id.

    Requires HTTP Basic Authentication.

    Status Codes:
        - 401: Need to authenticate.
        - 404: Event does not exist.
        - 200: Request successful.

    On 200, returns a JSON object with the following properties:
        - `displayname`: the display name of the event
        - `description`: the description of the event
        - `start`, `end`: the start and end datetimes of the event, as objects with the following integer attributes:
            - `year`, `month`, `day`
            - `hour`, `minute`
        - `venue`: the name of the event's venue, or null
        - `address`: the address of the event's venue
        - `coords`: the coordinates of the event's venue, with the following properties:
            - `lat`: latitude
            - `lon`: longitude
        - `attendees`: an array of users attending the event, each in the following format:
            - `id`: the user's ID
            - `username`: the user's username
            - `displayname`: the user's display name
        - `attending`: whether the logged-in user is attending the event
    """
    info = get_event_info(eventid)
    if info is None:
        abort(404)
    return info


def event_add_user(eventid: int, userid: int) -> Response:
    """
    Adds the user with the given id to the event with the given ID.
    
    Requires `g.userid` to be set (i.e., a function that calls it should be wrapped with `@authenticate`).

    If no errors are thrown, this will return a successful response (see docstring for `event_put()`).
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)

    with g.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events WHERE id = %s;", (eventid,))
        count, = cur.fetchone()
        if count == 0:
            abort(404)
        if userid != g.userid:
            abort(403)
        cur.execute("SELECT COUNT(*) FROM attendees WHERE userid = %s AND eventid = %s;", (g.userid, eventid))
        count, = cur.fetchone()
        if count == 0:
            cur.execute("INSERT INTO attendees(userid, eventid) VALUES (%s, %s);", (g.userid, eventid))
            return ("", 201)
        else:
            return ("", 204)

@bp.route("/<int:eventid>/user/<int:userid>", methods=["PUT"])
@authenticate
def event_user_put(eventid: int, userid: int) -> Response:
    """
    Adds the user with the given id to the event with the given ID.
    A user can only add themselves to any event.

    Requires HTTP Basic Authentication.

    Input: None

    Status Codes:
        - 401: Need to authenticate.
        - 404: Event does not exist.
        - 403: User is not authorized to add this user.
        - 201: User was successfully added to the event.
        - 204: User is already in event (still successful).
    
    The response will not contain any content.
    """
    return event_add_user(eventid, userid)


def event_remove_user(eventid: int, userid: int) -> Response:
    """
    Removes the given user from the given event.

    Requires `g.userid` to be set (i.e., a function that calls it should be wrapped with `@authenticate`).

    Returns a successful response (see docstring for `event_user_delete()`) if no errors are thrown.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        if g.userid != userid:
            cur.execute("SELECT host FROM events WHERE id = %s;", (eventid,))
            if cur.rowcount == 0:
                abort(404)
            host, = cur.fetchone()
            if g.userid != host:
                abort(403)
        cur.execute("DELETE FROM attendees WHERE userid = %s AND eventid = %s;", (userid, eventid))
        if cur.rowcount == 0:
            abort(404)
    return ("", 204)
        

@bp.route("/<int:eventid>/user/<int:userid>", methods=["DELETE"])
@authenticate
def event_user_delete(eventid: int, userid: int) -> Response:
    """
    Removes a user from an event.
    A user can remove themselves from any event they are attending, and a host can remove any user from their own event.

    Requires HTTP Basic Authentication.

    Input: None

    Status Codes:
        - 401: Need to authenticate.
        - 404: Event does not exist, or user is not attending event.
        - 403: User is not authorized to remove this user.
        - 204: Deletion was successful.
    
    The response will not contain any content.
    """
    return event_remove_user(eventid, userid)


def delete_event(eventid: int) -> Response:
    """
    Deletes the given event.

    Requires `g.userid` to be set (i.e., a function that calls it should be wrapped with `@authenticate`).

    Returns a successful response (see docstring for `event_delete()`) if no errors are thrown.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        cur.execute("SELECT host FROM events WHERE id = %s;", (eventid,))
        if cur.rowcount == 0:
            abort(404)
        host, = cur.fetchone()
        if host != g.userid:
            abort(403)
        cur.execute("DELETE FROM events WHERE id = %s;", (eventid,))
        return ("", 204)

@bp.route("/<int:eventid>", methods=["DELETE"])
@authenticate
def event_delete(eventid: int):
    """
    Deletes the event with the given ID.
    An event can only be deleted by its host.

    Requires HTTP Basic Authentication.

    Input: None

    Status Codes:
        - 401: Need to authenticate.
        - 404: Event does not exist.
        - 403: User is not authorized to delete this event.
        - 204: Deletion was successful.

    The response will not contain any content.
    """
    return delete_event(eventid)


def update_event_settings(eventid: int, **kwargs) -> Response:
    """
    Updates the settings for the given event to those in **kwargs.
    If a setting is listed in **kwargs, it will be updated. Otherwise, it will remain the same.

    Requires `g.userid` to be set.

    Returns either a successful or 422 response (see docstring for `event_patch()`) if no errors are thrown.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        cur.execute("SELECT host, displayname, start, end, location, description, embedding FROM events WHERE id = %s;", (eventid,))
        if cur.rowcount == 0:
            abort(404)
        host, displayname, start, end, location, description, embedding = cur.fetchone()
        if g.userid != host:
            abort(403)
        if "displayname" in kwargs:
            displayname = kwargs["displayname"]
        if "start" in kwargs:
            start = to_datetime(kwargs["start"])
        if "end" in kwargs:
            end = to_datetime(kwargs["end"], end=True)
        if "description" in kwargs:
            description = kwargs["description"]
        if "location" in kwargs:
            location = kwargs["location"]
        errors = validate_create_inputs(displayname, start, end, description, location)
        if errors:
            return (errors, 422)
        if "displayname" in kwargs or "description" in kwargs:
            assert isinstance(displayname, str)
            assert isinstance(description, str)
            placename, _, _ = get_place_info(location)
            embedding = emb.get_embedding(displayname.strip().replace("\n", " ") + " " + description.strip().replace("\n", " ") + (placename.strip().replace("\n", " ") if placename is not None else ""), engine="text-embedding-ada-002", user=str(g.userid))
        cur.execute("UPDATE events SET displayname = %s, start=%s, \"end\"=%s, description = %s, place = %s, embedding = %s WHERE id = %s;",
                    (displayname, start, end, description if description is not None else "", location, embedding, eventid))
        return ("", 204)

@bp.route("/<int:eventid>", methods=["PATCH"])
@authenticate
def event_patch(eventid: int):
    """
    Update the settings for the given event.
    Only the event's host can change its settings.

    Input: a JSON object with zero or more of the following objects:
        - `displayname`: the name of the event
        - `description`: a description of the event, or NULL
        - `start`, `end`: the start and end datetimes, with the following integer attributes:
            - `year`, `month`, `day`
            - `hour`, `minute` (optional)
        - `location`: the Google Place ID of the event's venue.
    
    The validation requirements are the same as those for the "/event" creation hook (see docstring for `create()`).
    
    Status Codes:
        - 401: Need to authenticate.
        - 415: Request body is not in JSON format.
        - 404: Event does not exist.
        - 403: User is not authorized to make changes.
        - 422: One or more of the setting changes are invalid.
        - 204: Changes were successful.
    
    A successful (204) response will contain no content.

    An invalid (422) response will contain a JSON array of errors in the following format:
        - `field`: the field in which the error occurred
        - `description`: a description of the error
    """
    if request.json is None:
        abort(415)
    if not isinstance(request.json, dict):
        abort(400)
    if "displayname" in request.json and not isinstance(request.json["displayname"], str):
        abort(400)
    if "start" in request.json:
        if not isinstance(request.json["start"], dict):
            abort(400)
        if "year" not in request.json["start"] or "month" not in request.json["start"] or "day" not in request.json["start"] \
            or ("minute" in request.json["start"] and "hour" not in request.json["start"]):
            abort(400)
    if "end" in request.json:
        if not isinstance(request.json["end"], dict):
            abort(400)
        if "year" not in request.json["end"] or "month" not in request.json["end"] or "day" not in request.json["end"] \
            or ("minute" in request.json["end"] and "hour" not in request.json["end"]):
            abort(400)
    try:
        if "start" in request.json:
            _ = int(request.json["start"]["year"])
            _ = int(request.json["start"]["month"])
            _ = int(request.json["start"]["day"])
            _ = int(request.json["start"].get("hour", 0))
            _ = int(request.json["start"].get("minute", 0))
        if "end" in request.json:
            _ = int(request.json["end"]["year"])
            _ = int(request.json["end"]["month"])
            _ = int(request.json["end"]["day"])
            _ = int(request.json["end"].get("hour", 0))
            _ = int(request.json["end"].get("minute", 0))
    except ValueError:
        abort(400)
    if "description" in request.json and request.json["description"] is not None \
        and not isinstance(request.json["description"], str):
        abort(400)
    if "location" not in request.json or not isinstance(request.json["location"], str):
        abort(400)
    if "eventid" in request.json:
        abort(400)
    return update_event_settings(eventid = eventid, **request.json)


def validate_message(text: str, **kwargs) -> Response | None:
    """
    Ensures the given message is valid.

    Returns an error JSON (as explained in the docstring for `chat_post()`) or `None` if the message is valid.
    """
    if len(text) == 0:
        return {"description": "Message cannot be empty."}
    if len(text) > 2000:
        return {"description": "Message must not be more than 2000 characters."}
    return None

def send_message(eventid: int, text: str, **kwargs) -> Response:
    """
    Sends a message with the given content to the event with the given id.

    Requires `g.userid` to be set (i.e., a function that calls it should be wrapped with `@authenticate`).

    Returns a successful JSON object (as explained in the docstring for `chat_post()`) if no errors are thrown.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM events WHERE id = %s;", (eventid,))
        count, = cur.fetchone()
        if count == 0:
            abort(404)
        cur.execute("SELECT COUNT(*) FROM attendees WHERE userid = %s AND eventid = %s;", (g.userid, eventid))
        count, = cur.fetchone()
        if count == 0:
            abort(403)
        for _ in range(100):
            try:
                cur.execute("INSERT INTO messages(eventid, sender, content) VALUES (%s, %s, %s) RETURNING time;", (eventid, g.userid, text))
                time, = cur.fetchone()
                return {"time": time}
            except psycopg.errors.UniqueViolation:
                sleep(0.01)
        print("Error: server timeout while posting message to event %d." % eventid)
        abort(500)

@bp.route("/<int:eventid>/chat", methods=["POST"])
@authenticate
def chat_post(eventid: int) -> Response:
    """
    Post a new message to the event's chat.
    Requires the logged-in user to be attending the event.

    Requires HTTP Basic Authentication.

    Input: a JSON object containing the following properties:
        - `text`: the text of the message

    Status Codes:
        - 401: Need to authenticate.
        - 415: Request body is not of JSON type.
        - 400: Request is not in the right format.
        - 422: Message was invalid.
        - 404: Event does not exist.
        - 403: User is not authorized to make this request.
        - 500: Server timed out while sending message.
        - 200: Request succeeded.
    
    On a 200 (successful) status code, a JSON object with the following values will be returned.
        - `time`: the unique (per-event) timestamp of the message.    
    
    An invalid (422) response will contain a JSON object with the following properties:
        - `description`: a description of the error
    """
    if not isinstance(request.json, dict):
        abort(400)
    if "text" not in request.json or not isinstance(request.json["text"], str):
        abort(400)
    if "eventid" in request.json:
        abort(400)
    errors = validate_message(eventid = eventid, **request.json)
    if errors is not None:
        return errors
    return send_message(eventid = eventid, **request.json)


def most_recent_messages(since: int | None, eventid: int, n = 20) -> Response:
    """
    Gets the most recent messages in the given event before the given timestamp.

    Requires `g.userid` to be set (i.e., a function that calls this must be wrapped in `@authenticate`).

    Returns a list of at most `n` messages (see the docstring for `chat_get()`) if no error is thrown.
    """
    messages = []

    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        cur.execute("SELECT host FROM events WHERE id = %s;", (eventid,))
        if cur.rowcount == 0:
            abort(404)
        host, = cur.fetchone()
        if g.userid != host:
            cur.execute("SELECT COUNT(*) FROM attendees WHERE userid = %s AND eventid = %s;", (g.userid, eventid))
            count, = cur.fetchone()
            if count == 0:
                abort(403)
        if since is None:
            cur.execute("SELECT messages.time, messages.content, users.id, users.username, users.displayname \
                    FROM (messages LEFT JOIN users ON messages.sender = users.id) \
                    WHERE messages.eventid = %s \
                    ORDER BY messages.time DESC LIMIT %s;", (eventid, n))
        else:
            cur.execute("SELECT messages.time, messages.content, users.id, users.username, users.displayname \
                    FROM (messages LEFT JOIN users ON messages.sender = users.id) \
                    WHERE messages.eventid = %s AND messages.time < %s\
                    ORDER BY messages.time DESC LIMIT %s;", (eventid, since, n))
        for row in cur:
            time, text, uid, uname, dname = row
            msg = {"time": time, "text": text, "sender": None}
            if uid is not None:
                msg["sender"] = {"id": uid, "username": uname, "displayname": dname}
            messages.append(msg)
    return messages

@bp.route("/<int:eventid>/chat", methods=["GET"], defaults={"time": None})
@bp.route("/<int:eventid>/chat/<int:time>")
@authenticate
def chat_get(eventid: int, time: int | None) -> Response:
    """
    Gets the most recent 20 chat messages.
    If a time is provided, it gets the most recent 20 chat messages before that timestamp.
    Only a user who is attending an event or the host of the event can view messages in the chat.

    Requires HTTP Basic Authentication.

    Input: None

    Status Codes:
        - 401: Need to authenticate.
        - 404: Event does not exist.
        - 403: User is not authorized to make this request.
        - 200: Request successful.
    
    On a 200 (successful) status code, a JSON array of messages will be returned,
    with the most recent messages first, each with the following properties:
        - `time`: the unique (per-event) timestamp of the message
        - `text`: the text of the message
        - `sender`: an object containing information about the sender, or NULL if the user was deleted
            - `id`: the user's unique ID
            - `username`: the user's username
            - `displayname`: the user's display name
    """
    return most_recent_messages(since=time, eventid=eventid)


def delete_message(eventid: int, time: int) -> Response:
    """
    Delete the message with the given timestamp from the event with the given ID.

    Requires `g.userid` to be set (i.e., a function that calls this must be wrapped in `@authenticate`).

    Returns an empty successful Response (see the docstring for `chat_delete()`) if no errors are thrown.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    with g.conn.cursor() as cur:
        cur.execute("SELECT host FROM events WHERE id = %s;", (eventid,))
        if cur.rowcount == 0:
            abort(404)
        host, = cur.fetchone()
        cur.execute("SELECT sender FROM messages WHERE eventid = %s AND time = %s;", (eventid, time))
        if cur.rowcount == 0:
            abort(404)
        sender, = cur.fetchone()
        if g.userid != host and (sender is None or g.userid != sender):
            abort(403)
        cur.execute("DELETE FROM messages WHERE eventid = %s AND time = %s;", (eventid, time))
        return ("", 204)

@bp.route("/<int:eventid>/chat/<int:time>", methods=["DELETE"])
@authenticate
def chat_delete(eventid: int, time: int):
    """
    Delete the message with the given unique (per-event) timestamp from the event with the given ID.
    Only the host and the user who sent the message are authorized to send this request.

    Requires HTTP Basic Authentication.

    Input: None

    Status Codes:
        - 401: Need to authenticate.
        - 404: Event or message does not exist.
        - 403: User is not authorized to delete this message.
        - 204: Request was successful.
    
    Nothing is returned in the response body.
    """
    return delete_message(eventid, time)


def get_events_by_location(location: Point, distance: float, earliest: datetime | None = None, latest: datetime | None = None) -> list:
    """
    List all events not more than `distance` miles from `location`, sorted by distance in ascending order.

    If `earliest` or `latest` are given, the results are constrained to these boundaries.
    """
    assert isinstance(g.conn, psycopg.Connection)
    with g.conn.cursor() as cur:
        events = []
        cur.execute("SELECT events.id, events.displayname, events.start, events.\"end\", events.description, places.name, places.address, places.coords FROM (events JOIN places ON events.place = places.id);")
        for row in cur:
            id, dname, start, end, description, pname, addr, coords = row
            if earliest is not None and end < earliest:
                continue
            if latest is not None and start > latest:
                continue
            assert isinstance(coords, Point)
            dist = location.distanceto(coords)
            if dist.miles <= distance:
                events.append({"id": id, "displayname": dname, "distance": dist.miles, "coords": {"lat": coords.lat, "lon": coords.lon}, "start": from_datetime(start), "end": from_datetime(end), "venue": pname, "address": addr, "description": description})
        events.sort(key=lambda x: x["distance"])
        return events

def get_events_by_keyword(query: str, earliest: datetime | None = None, latest: datetime | None = None, location: Point | None = None, distance: float = float("inf")):
    """
    List all events not more than `distance` miles from `location`, sorted by relevance to the given query/keyword.

    If `earliest` or `latest` are given, the results are constrained to these boundaries.
    """
    assert isinstance(g.conn, psycopg.Connection)
    assert isinstance(g.userid, int)
    if not query.strip():
        abort(400)
    with g.conn.cursor() as cur:
        events = []
        cur.execute("SELECT events.id, events.displayname, events.embedding, events.start, events.\"end\", events.description, places.name, places.address, places.coords FROM (events JOIN places ON events.place = places.id);")
        for row in cur:
            id, dname, evemb, start, end, description, pname, addr, coords = row
            if earliest is not None and end < earliest:
                continue
            if latest is not None and start > latest:
                continue
            if location is not None and location.distanceto(coords).miles > distance:
                continue
            events.append({"id": id, "displayname": dname, "description": description, "venue": pname, "address": addr, "coords": coords, "embedding": evemb, "distance": None, "start": from_datetime(start), "end": from_datetime(end)})
            if location is not None:
                events[-1]["distance"] = location.distanceto(coords).miles
        if len(events) >= 2:
            embedding = emb.get_embedding(query.strip().replace("\n", " "), engine="text-embedding-ada-002", user=str(g.userid))
            events.sort(key=lambda x: -emb.cosine_similarity(embedding, x["embedding"]))
        for i in range(len(events)):
            events[i].pop("embedding")
        return events

@bp.route("/search", methods=["GET"])
@authenticate
def event_search():
    """
    Search for an event given the user's coordinates and/or a search query.

    Input: URL query arguments:
        - `query` (optional): the search query
        - `ey`, `em`, `ed` (optional): the earliest date for which to return results
        - `ly`, `lm`, `ld` (optional): the latest date for which to return results
        - `lat` (optional): the user's latitude, as a decimal
        - `lon` (optional): the user's longitude, as a decimal
        - `radius` (optional): the maximum search radius, in miles
    
    Input Requirements:
        - Either `query` or `lat` and `lon` must be included (or all three).
        - If `query` is included, it must not be empty.
        - If `radius` is included without `lat` and `lon`, it will be ignored.
        - `radius` must be nonnegative if it is included.
    
    Status Codes:
        - 401: Need to authenticate.
        - 400: Did not include correct arguments.
        - 200: Request successful.
    
    Output: a JSON array of events, in the following form, sorted by distance in ascending order:
        - `id`: the event's ID
        - `displayname`: the display name of the event
        - `description`: the description of the event
        - `start`, `end`: the start and end datetimes of the event, as objects with the following integer attributes:
            - `year`, `month`, `day`
            - `hour`, `minute`
        - `venue`: the name of the event's venue, or null
        - `address`: the address of the event's venue
        - `coords`: the coordinates of the event's venue, with the following properties:
            - `lat`: latitude
            - `lon`: longitude
        - `distance`: if `lat` and `lon` were provided, the distance, in miles, to the event
    """
    startdt = enddt = None
    if "ey" in request.args:
        try:
            startdt = to_datetime({"year": request.args["ey"], "month": request.args.get("em", 1), "day": 1 if "em" not in request.args else request.args.get("ed", 1)})
        except ValueError:
            abort(400)
    if "ly" in request.args:
        try:
            year = int(request.args["ly"])
            month = int(request.args.get("lm", 12))
            day = None
            if "lm" in request.args and "ld" in request.args:
                day = int(request.args["ld"])
            else:
                nextday = None
                if month == 12:
                    day = 31
                else:
                    nextday = datetime(year, month + 1, 1)
                    day = (nextday - timedelta(days=1)).day
            enddt = to_datetime({"year": year, "month": month, "day": day}, end=True)
        except ValueError:
            abort(400)
    if "lat" in request.args and "lon" in request.args:
        try:
            lat = float(request.args["lat"])
            lon = float(request.args["lon"])
            radius = float(request.args.get("radius", default="inf"))
            if radius < 0:
                abort(400)
            if "query" in request.args:
                return get_events_by_keyword(request.args["query"], startdt, enddt, Point(lat, lon), radius)
            else:
                return get_events_by_location(Point(lat, lon), radius, startdt, enddt)
        except ValueError:
            abort(400)
    elif "query" in request.args:
        return get_events_by_keyword(request.args["query"], startdt, enddt)
    else:
        abort(400)