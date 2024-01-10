import os  
import datetime 
import config

from flask import Flask
from flask import request
from flask_cors import CORS
from flask_caching import Cache

import openai 
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex
from llama_index import Document
from typing import List
from pydantic import BaseModel

from metaphor_python import Metaphor

# Set up flask app
app_config = {
    "DEBUG": True,          
    "CACHE_TYPE": "SimpleCache", 
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__)
app.config.from_mapping(app_config)
cache = Cache(app)
CORS(app)

metaphor = Metaphor(config.METAPHOR_API_KEY)
openai.api_key = config.OPENAI_API_KEY 

class Event(BaseModel):
    """Data model for an event"""
    id: str
    url: str
    title: str
    year: int
    month: int
    day: int
    extra_info: str

class EventList(BaseModel):
    '''Data model for list of events'''
    events: List[Event]

@cache.cached(timeout=50, key_prefix='events')
@app.route("/event", methods=["GET"])
def get_event_details():
    '''
    Get the details (especailly year, month, day, and extra_info) of an event given it's id
    '''
    id = request.args.get('id')
    [content] = metaphor.get_contents([id]).contents

    document = Document(
        text="[Document] " + content.extract,
        metadata={
            'id': content.id,
            'url': content.url,
            'title': content.title
        }
    )
    index = VectorStoreIndex.from_documents([document])
    query_engine = index.as_query_engine(response_mode="tree_summarize", output_cls=Event)

    query = '''
    pretend to be an expert assistant that helps people schedule events. You are given an document
    that each represents the parsed text information of an event posted on a website.

    Create a valid JSON array of objects from the provided information. The returned result should follow this format:

    {
        id: "the id field from the given document",
        url:  "the id field from the given document",
        title: "the title field from the given document",
        year: the year that this event is taking place in integer format, if cannot find this information, use the current year,
        month: the month that this event is taking place in integer format, if cannot find this information, use the current month,
        day: the dat in the month that this event is taking place in integer format,
        extra_info: "extra information that you think is useful to the user"
    }

    The JSON object:
    '''
    event_obj = query_engine.query(query)
    res = {
        "id": event_obj.id,
        "url": event_obj.url,
        "title": event_obj.title,
        "year": event_obj.year,
        "month": event_obj.month,
        "day": event_obj.day,
        "extra_info": event_obj.extra_info
    }
    return res, 200

@app.route("/", methods=["GET"])
def get_events():
    '''
    Get a list of events from metaphor API and convert to proper json format for the frontend
    '''

    # Get a list of events from metaphor API
    location = request.args.get('location')
    try: 
        today = datetime.date.today()
        first = today.replace(day=1)
        last_month = first - datetime.timedelta(days=1)

        response = metaphor.search(
            "Check out this exciting recent event happening in " + location,
            num_results=5,
            start_crawl_date=last_month.isoformat(),
            use_autoprompt=True,
        )
    except Exception as e:
        print(e)
        return "Failed to get search results from Metaphor", 500

    return response.results, 200


@app.route("/rec", methods=["POST"])
def get_similar_links():
    '''
    return a list of similar links by calling the Find Similar route
    '''
    data = request.get_json()
    res = metaphor.find_similar(data["url"]).results

    return res, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)