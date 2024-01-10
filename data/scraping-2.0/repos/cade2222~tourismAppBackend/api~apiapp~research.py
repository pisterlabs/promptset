from flask import request, g, Blueprint, abort
from .auth import authenticate
from .location import Point
from datetime import datetime
from openai import embeddings_utils as emb
from typing import Any
import psycopg

bp = Blueprint("research", __name__, url_prefix="/research")

@bp.route("/placetypes", methods=["GET"])
@authenticate
def get_place_types():
    """
    Returns a JSON list of possible place types.
    """
    assert isinstance(g.conn, psycopg.Connection)
    with g.conn.cursor() as cur:
        types = []
        cur.execute("SELECT DISTINCT type FROM placetypes;")
        for row in cur:
            t, = row
            types.append(t)
        return types

def get_event_name(eventid: int) -> str | None:
    assert isinstance(g.conn, psycopg.Connection)
    with g.conn.cursor() as cur:
        cur.execute("SELECT displayname FROM events WHERE id = %s;", (eventid,))
        if cur.rowcount == 0:
            return None
        return cur.fetchone()[0]

def get_events(eventquery: str | None, startdate: datetime | None, enddate: datetime | None, eventlocation: tuple[Point, float] | None) -> list[dict[str, Any]]:
    THRESHOLD = 0.85
    assert isinstance(g.conn, psycopg.Connection)
    with g.conn.cursor() as cur:
        events = []
        cur.execute("SELECT events.id, events.displayname, events.start, events.\"end\", events.embedding, places.coords FROM (events JOIN places ON events.place = places.id);")
        for row in cur:
            eid, ename, estart, eend, eemb, ploc = row
            if enddate is not None and estart > enddate:
                continue
            if startdate is not None and eend < startdate:
                continue
            if eventlocation is not None:
                coords, rad = eventlocation
                if coords.distanceto(ploc).miles > rad:
                    continue
            if eventquery is not None and eventquery.strip():
                qemb = emb.get_embedding(eventquery.strip().replace("\n", " "))
                if emb.cosine_similarity(qemb, eemb) < THRESHOLD:
                    continue
            events.append({"id": eid, "displayname": ename})
        return events

def get_places(events: list[dict[str, Any]], placetype: str | None, placelocation: tuple[Point, float] | None) -> list[tuple[int, str, Point]]:
    from sys import stderr
    if not events:
        return []
    assert isinstance(g.conn, psycopg.Connection)
    with g.conn.cursor() as cur:
        places = []
        cur.execute("SELECT places.id, places.name, places.coords FROM (locations JOIN places ON locations.placeid = places.id) WHERE locations.eventid = ANY(%s);", (list(map(lambda x: x["id"], events)),))
        result = cur.fetchall()
        for row in result:
            pid, pname, ploc = row
            assert isinstance(ploc, Point)
            if placelocation is not None and placelocation[0].distanceto(ploc).miles > placelocation[1]:
                continue
            cur.execute("SELECT type FROM placetypes WHERE id = %s;", (pid,))
            types = [i[0] for i in cur.fetchall()]
            if placetype is not None and placetype not in types:
                continue
            places.append({"id": pid, "name": pname, "coords": {"lat": ploc.lat, "lon": ploc.lon}, "types": types})
        return places

def get_location_data(events: list[dict[str, Any]], places: list[dict[str, Any]]):
    assert isinstance(g.conn, psycopg.Connection)
    with g.conn.cursor() as cur:
        eids = list(map(lambda x: x["id"], events))
        pids = list(map(lambda x: x["id"], places))
        locs = {i: {j: 0 for j in pids} for i in eids}
        if eids and pids:
            cur.execute("SELECT eventid, placeid, visits FROM locations WHERE eventid = ANY(%s) AND placeid = ANY(%s);", (eids, pids))
            for row in cur:
                eid, pid, visits = row
                locs[eid][pid] = visits
        return {"events": events, "places": places, "visits": locs}

@bp.route("", methods=["GET"])
@authenticate
def get_research_info():
    """
    Returns collected location data for a given event or set of events.

    Inputs (given as URL parameters, all optional):
        - `eventid`: the ID of the event, if only one is being examined
        - `start`: the earliest date for which to accept events
        - `end`: the latest date for which to accept events
        - `eventquery`: a query on which to limit events (must not be empty)
        - `eventlat`, `eventlon`, `eventrad`: the coordinates and maximum search radius on which to restrict events
        - `placetype`: limit places by type (to get a list of available types, query "/research/placetypes")
        - `placelat`, `placelon`, `placerad`: the coordinates and maximum search radius on which to restrict places
    
    Outputs a JSON object:
        - `events`: a list of returned events in the following format:
            - `id`: the event's ID
            - `displayname`: the event's name
        - `places`: a list of returned places in the following format:
            - `id`: the place's ID
            - `name`: the place's name
            - `coords`: the place's location, in the following format:
                - `lat`, `lon`: the latitude and longitude of the place, in degrees
            - `types`: a list of the place's types
        - `visits`: the number of times people in a given event visited a given place
            - indexed first on event ID, then on place ID (i.e., access `visits[eventid][placeid]`)
            - only includes the events and places listed in `events` and `places`
    """
    startdate = enddate = eventid = eventlocation = placelocation = None
    try:
        if "start" in request.args:
            startdate = datetime.fromisoformat(request.args["startdate"])
        if "end" in request.args:
            enddate = datetime.fromisoformat(request.args["end"])
            enddate.hour = 23
            enddate.minute = 59
        if "eventid" in request.args:
            eventid = int(request.args["eventid"])
        if "eventlat" in request.args and "eventlon" in request.args and "eventradius" in request.args:
            lat = float(request.args["eventlat"])
            lon = float(request.args["eventlon"])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                location = Point(lat, lon)
            radius = float(request.args["eventradius"])
            eventlocation = (location, radius)
        if "placelat" in request.args and "placelon" in request.args and "placeradius" in request.args:
            lat = float(request.args["placelat"])
            lon = float(request.args["placelon"])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                location = Point(lat, lon)
            radius = float(request.args["placeradius"])
            placelocation = (location, radius)
    except ValueError:
        abort(400)
    eventquery = request.args.get("eventquery")
    placetype = request.args.get("placetype")
    
    eventlist = []
    if eventid is not None:
        name = get_event_name(eventid)
        if name is not None:
            eventlist.append((eventid, name))
    else:
        eventlist = get_events(eventquery, startdate, enddate, eventlocation)
    placelist = get_places(eventlist, placetype, placelocation)
    return get_location_data(eventlist, placelist)
