from pydantic import BaseModel, Field

from airflow.hooks.postgres_hook import PostgresHook
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from tqdm import tqdm


response_json_format = """{
"locations": [
{
"name":
"context":
"coordinates": {
"lat":
"long":
}
}
],
"time": {
"description":
"start_year":
"end_year":
},
"categories": [
{
"category_name":
"description":
}
]
}"""


class Coordinate(BaseModel):
    lat: Optional[float] = Field(..., example=52.520008)
    long: Optional[float] = Field(..., example=13.404954)


class Location(BaseModel):
    name: str = Field(..., example="Berlin")
    context: Optional[str] = Field(..., example="Hauptstadt von Deutschland")
    coordinates: Optional[Coordinate] = Field(..., example={
        "lat": 52.520008, "long": 13.404954})

    @validator('coordinates', pre=True)
    def empty_dict_to_none(cls, v):
        if v == {}:
            return None
        return v


class Time(BaseModel):
    description: str = Field(...,
                             example="Die Geschichte spielt im 19. Jahrhundert.")
    start_year: Optional[float] = Field(..., example=1800,
                                        description="Das Jahr, in dem die Geschichte beginnt.")
    end_year: Optional[float] = Field(..., example=1900,
                                      description="Das Jahr, in dem die Geschichte endet.")


class Category(BaseModel):
    category_name: str = Field(..., example="Krieg")
    description: str = Field(
        ..., example="Der Hauptakteur ist ein Soldat im 2. Weltkrieg.")


class Response(BaseModel):
    locations: List[Location] = Field(
        description="Eine Liste von Orten, die in der Geschichte vorkommen.")
    time: Time = Field(description="Die Zeit, in der die Geschichte spielt.")
    categories: List[Category] = Field(
        description="Eine Liste von Kategorien, in die die Geschichte eingeordnet werden kann.")


def get_response_parser():
    parser = PydanticOutputParser(pydantic_object=Response)
    output_fixer = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

    return output_fixer


def get_chain():
    system_message = """
    Du bist ein Assistent der nur JSON spricht. Jede Antwort muss ein korrektes JSON sein, welches die folgenden Felder enthält: locations, time, categories.
    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_message)

    prompt = HumanMessagePromptTemplate.from_template(
        input_variables=["summary", "json_format"],
        template="""Der Text beinhaltet den Titel und eine Zusammenfassung einer Geschichts-Podcast Episode. Schreib mir für den Text alle enthaltenen Orte, die Zeit in der die Geschichte aus der Episode spielt und die Kategorien, in welche die Geschichte eingeordnet werden kann. Schreibe für die Orte den Namen ohne Zusatz, den Kontext, in dem Sie besprochen werden und die genauen Koordinaten als numerische Werte ohne Angabe von Maßeinheit oder Himmelsrichtung. Für die Zeit schreibe eine textliche Beschreibung, sowie das Jahr in dem die Geschichte beginnt und endet als Ganzzahl. Für die Kategorien, brauche ich einen Namen und eine kurze Beschreibung und Einordnung der Geschichte in die Kategorie. Das Wort Geschichte darf keine eigene Kategorie sein.
        Gebe die Antwort im Format:
        {json_format}
        
        {summary}
        """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, prompt])
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)

    return LLMChain(llm=llm, prompt=chat_prompt)


def gpt_extract():
    openai_chain = get_chain()

    postgres_sql = PostgresHook(
        postgres_conn_id='postgres_be', schema='kag')
    episode_df = postgres_sql.get_pandas_df("""
        SELECT id, key, summary, title, subtitle
        FROM episodes_target
        WHERE status = 'preprocessed' AND is_gpt_integrated = false
        """)

    episode_df['text'] = episode_df['title'] + '. ' + \
        episode_df['subtitle'] + '. ' + episode_df['summary']
    episode_df = episode_df[['id', 'key', 'text']]

    parser = get_response_parser()

    for entry in tqdm(episode_df.itertuples(), total=len(episode_df)):
        print(f"Requesting {entry.key}")
        response = openai_chain.run(
            summary=entry.text, json_format=response_json_format)

        validated = parser.parse(response)

        time = validated.time
        topics = validated.categories
        locations = validated.locations

        for topic in topics:
            postgres_sql.run("""
                INSERT INTO topics (name, status, origin)
                VALUES (%(name)s, %(status)s, %(origin)s)
                ON CONFLICT (name) DO NOTHING
                """, parameters={"name": topic.category_name, "status": "active", "origin": "gpt-3.5-turbo"})
            postgres_sql.run("""
                INSERT INTO episodes_topics (episode_id, topic_id, context)
                VALUES (%(episode_id)s, (SELECT id FROM topics WHERE name = %(name)s), %(context)s)
                ON CONFLICT (episode_id, topic_id) DO NOTHING
                """, parameters={"episode_id": entry.id, "name": topic.category_name, "context": topic.description, "origin": "gpt-3.5-turbo"})

        for location in locations:
            postgres_sql.run("""
                INSERT INTO locations (name, status, origin)
                VALUES (%(name)s, %(status)s, %(origin)s)
                ON CONFLICT (name) DO NOTHING
                """, parameters={"name": location.name, "status": "active", "origin": "gpt-3.5-turbo"})
            postgres_sql.run("""
                INSERT INTO episodes_locations (episode_id, location_id, context)
                VALUES (%(episode_id)s, (SELECT id FROM locations WHERE name = %(name)s), %(context)s)
                ON CONFLICT (episode_id, location_id) DO UPDATE SET context = %(context)s
                """, parameters={"episode_id": entry.id, "name": location.name, "context": location.context, "origin": "gpt-3.5-turbo"})
            if location.coordinates is not None:
                # Check if coordinate for location generated by GPT already exists
                coordinate_exists = postgres_sql.get_pandas_df("""
                    SELECT * FROM coordinates
                    WHERE location_id = (SELECT id FROM locations WHERE name = %(name)s)
                    AND origin = %(origin)s
                    """, parameters={"name": location.name, "origin": "gpt-3.5-turbo"}).shape[0] > 0

                if not coordinate_exists and location.coordinates.lat is not None and location.coordinates.long is not None:
                    postgres_sql.run("""
                        INSERT INTO coordinates (location_id, latitude, longitude, status, origin)
                        VALUES ((SELECT id FROM locations WHERE name = %(name)s), %(latitude)s, %(longitude)s, %(status)s, %(origin)s)
                        ON CONFLICT (location_id, latitude, longitude) DO NOTHING
                        """, parameters={"name": location.name, "latitude": location.coordinates.lat, "longitude": location.coordinates.long, "status": "active", "origin": "gpt-3.5-turbo"})

        postgres_sql.run("""
            UPDATE episodes_target
            SET story_time_start = %(start_year)s, story_time_end = %(end_year)s, story_time_description = %(description)s, is_gpt_integrated = true
            WHERE id = %(id)s
            """, parameters={"id": entry.id, "start_year": time.start_year, "end_year": time.end_year, "description": time.description})
