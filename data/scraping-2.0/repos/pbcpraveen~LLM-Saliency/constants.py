from enum import Enum

import openai
from dotenv import load_dotenv
import os


load_dotenv('api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_COLUMN = 'prompt'
GPT_4_RESPONSE = 'GPT 4 Response'
INDEX_COLUMN = "index"
PROMPT_INDEX_COLUMN = "prompt_index"
GROUND_TRUTH = "ground_truth"
ICL_PROMPT_COLUMN = "icl_prompt"


ENTITY = "entity"
CONTEXTUALISING_ATTRIBUTES = "contextualising_attributes"
TARGET_ATTRIBUTES = "target_attributes"
VERIFIED = "verified"

# Sources
WIKIBIO = "wiki_bio"
NOBEL_LAUREATES_DATASET = ("https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/nobel-prize-laureates/exports/"
                       "json?lang=en&timezone=America%2FLos_Angeles")
NOBEL_LAUREATES = "nobel_laureates"
MOVIE_DATASET = ("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
MOVIE = "movie"

COUNTRY_DATASET = ("https://raw.githubusercontent.com/bastianherre/global-leader-ideologies/"
                   "main/global_leader_ideologies.csv")

ATTRIBUTE_VERIFIED = "attribute_verified"

VERIFIED_RECORDS = {
    WIKIBIO: "wikibio_verified.pickle",
    NOBEL_LAUREATES: "nobel_laureates_verified.pickle",
    MOVIE: "movies_verified.pickle"
}
class ConceptClass(Enum):
    PLACE = "place"
    PERSON_NAME = "name"
    YEAR = "year"


class EntityClass(Enum):
    PERSON = "person"
    NOBEL_LAUREATES = "nobel_laureates"
    MOVIE = "movie"


class Attribute(Enum):
    NAME = "name"
    NATIONALITY = "nationality"
    OCCUPATION = "occupation"
    BIRTH_DATE = "birth_date"
    DEATH_DATE = "death_date"
    BIRTH_PLACE = "birth_place"
    DEATH_PLACE = "death_place"
    MOTIVATION_NOBEL = "motivation"
    CATEGORY_NOBEL = "category"
    BIRTH_DATE_NOBEL = "born"
    DEATH_DATE_NOBEL = "died"
    YEAR = "year"
    BIRTH_CITY = "borncity"
    DEATH_CITY = "diedcity"
    WORK_CITY = "city"
    FIRST_NAME = "firstname"
    SURNAME = "surname"
    MOVIE_TITLE = "Series_Title"
    MOVIE_DESCRIPTION = "Overview"
    RELEASE_YEAR_MOVIE = "Released_Year"
    CERTIFICATE_MOVIE = "Certificate"
    GENRE_MOVIE = "Genre"
    IMDB_RATING_MOVIE = "IMDB_Rating"
    VOTES_COUNT_MOVIE = "No_of_Votes"
    DIRECTOR_MOVIE = "Director"
    STAR1_MOVIE = "Star1"
    STAR2_MOVIE = "Star2"
    STAR3_MOVIE = "Star3"
    STAR4_MOVIE = "Star4"
    COUNTRY_NAME = "country_name"
    LEADER_NAME = "leader"
    LEADER_POSITION = "leader_position"




metadata = {
    WIKIBIO: {
        ENTITY: EntityClass.PERSON.value,
        CONTEXTUALISING_ATTRIBUTES: [
            Attribute.NAME.value,
            Attribute.NATIONALITY.value,
            Attribute.OCCUPATION.value
        ],
        TARGET_ATTRIBUTES: {
            ConceptClass.YEAR.value: [Attribute.BIRTH_DATE.value, Attribute.DEATH_DATE.value],
            ConceptClass.PLACE.value: [Attribute.BIRTH_PLACE.value, Attribute.DEATH_PLACE.value]
        }
    },
    NOBEL_LAUREATES: {
        ENTITY: EntityClass.NOBEL_LAUREATES.value,
        CONTEXTUALISING_ATTRIBUTES: [
            Attribute.FIRST_NAME.value,
            Attribute.SURNAME.value,
            Attribute.MOTIVATION_NOBEL.value,
            Attribute.CATEGORY_NOBEL.value
        ],
        TARGET_ATTRIBUTES: {
            ConceptClass.YEAR.value: [
                Attribute.BIRTH_DATE_NOBEL.value,
                Attribute.DEATH_DATE_NOBEL.value,
                Attribute.YEAR.value
            ],
            ConceptClass.PLACE.value: [
                Attribute.BIRTH_CITY.value,
                Attribute.DEATH_CITY.value,
                Attribute.WORK_CITY.value
            ]
        }
    },
    MOVIE: {
        ENTITY: EntityClass.MOVIE.value,
        CONTEXTUALISING_ATTRIBUTES: [
            Attribute.MOVIE_TITLE.value,
            Attribute.RELEASE_YEAR_MOVIE.value,
            Attribute.GENRE_MOVIE.value,
            Attribute.CERTIFICATE_MOVIE.value
        ],
        TARGET_ATTRIBUTES: {
            ConceptClass.PERSON_NAME.value: [Attribute.DIRECTOR_MOVIE.value,
                Attribute.STAR1_MOVIE.value,
                Attribute.STAR2_MOVIE.value,
                Attribute.STAR3_MOVIE.value,
                Attribute.STAR4_MOVIE.value]
        }
    }
}
