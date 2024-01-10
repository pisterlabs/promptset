import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from er_schema_normalization.er_types import *
from er_schema_normalization.helpers import run

from data_collection.scraper import generate_scraped_urls
from file_processing.create_er_csv import generate_smart_data

def create_demo_schema():
    """
    Generate and save to a file the set of normalized tables corresponding
    to a weather-related ER schema.
    """

    # Attributes
    WaterYear = Attribute("water_year", int)
    CalendarYear = Attribute("calendar_year", int)
    RelativeHumidity = Attribute("sme2_rh", float)

    # Entities
    Weather = Entity("weather")

    # Relationships (entity-attribute)
    WeatherWaterYear = Relationship("waterYear", RelationshipType.ONE_TO_ONE, Weather, WaterYear)
    WeatherCalendarYear = Relationship("calendarYear", RelationshipType.ONE_TO_ONE, Weather, CalendarYear)
    WeatherRelativeHumidity = Relationship("relativeHumidity", RelationshipType.ONE_TO_ONE, Weather, RelativeHumidity)

    schema = ERSchema(
        vertices=[Weather, WaterYear, CalendarYear, RelativeHumidity],
        edges=[WeatherWaterYear, WeatherCalendarYear, WeatherRelativeHumidity]
    )

    output_dir, schema_filename = run('weather', schema)
    return output_dir, schema_filename


def get_schema_headers(schema_file):
    """
    Currently only supports one normalized table - return its column headers
    """

    with open(schema_file, 'rb', encoding='utf-8-sig') as file:
        schema = json.load(file)

    if schema:
        first_table = next(iter(schema))
        col_headers = list(schema[first_table].keys())

        default_pk = [elt for elt in col_headers if elt.endswith("_id")]
        non_default_pk = [elt for elt in col_headers if not elt.endswith("_id")]

        assert len(default_pk) < 2 # should only have 0-1 default primary keys per table

        return {
            'default_pk': default_pk,
            'non_default_pk': non_default_pk
        }


def get_scraper_topic_no_gpt(table_headers):
    """
    Given the column headers from our normalized tables as input,
    generate a topic for the scraper to search on
    """

    return " ".join(table_headers)


def get_scraper_topic(table_headers, n_words=6):
    """
    Given the column headers from our normalized tables as input, use OpenAI to
    generate a topic for the scraper to search on
    """

    client = openai.OpenAI(api_key = OPENAI_API_KEY)
    gpt_input = " ".join(table_headers)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
        {"role": "system",
        "content": "Given this database schema:\n" + gpt_input + f"\n\nGenerate an on-topic phrase for the database that is less than {n_words} words."}
        ],
        temperature=0,
        max_tokens=256
    )

    return response.choices[0].message.content


def main():
    print("=" * 80 + "\n")
    output_dir, schema_file = create_demo_schema()
    schema_headers = get_schema_headers(schema_file)
    print("Schema headers:", schema_headers)
    print()

    topic = get_scraper_topic(schema_headers['non_default_pk'])
    print("Scraper topic:", topic)
    print()

    generate_scraped_urls(topic)
    generate_smart_data(output_dir, schema_headers, "../data_collection/links.txt")
    print("=" * 80 + "\n")
    # print("=" * 80 + "\n")
    # output_dir, schema_file = create_demo_schema()
    # schema_headers = get_schema_headers(schema_file)
    # print("Schema headers:", schema_headers)
    # print()

    # n_words = 6
    # found_with_gpt = False
    # while n_words > 1:
    #     topic = get_scraper_topic(schema_headers['non_default_pk'], n_words)
    #     generate_scraped_urls(topic)
    #     print("Scraper topic:", topic)
    #     print()
    #     try:
    #         generate_smart_data(output_dir, schema_headers, "../data_collection/links.txt")
    #         found_with_gpt = True
    #         break
    #     except:
    #         n_words -= 1
    #         print("Trying again with fewer words...")
    #         print()

    # if not found_with_gpt:
    #     print("Could not find a match with GPT-4. Trying without GPT-4..\n")
    #     topic = get_scraper_topic_no_gpt(schema_headers['non_default_pk'])
    #     print("Scraper topic:", topic)
    #     print()
    #     generate_scraped_urls(topic)
    #     generate_smart_data(output_dir, schema_headers, "../data_collection/links.txt")

    # print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
