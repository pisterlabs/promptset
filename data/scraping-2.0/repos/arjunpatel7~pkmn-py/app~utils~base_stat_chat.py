import argparse
from .calculations import read_in_pokemon, extract_stat
from langchain.llms import Cohere
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from cohere.error import CohereAPIError
import os
import streamlit as st
import cohere

# a chatbot for doing natural language querying on a dataset of pokemon stats

# we will use Langchain, Cohere, chains to accomplish this

# this will be a command line script that accepts the query and returns the result


def preprocess_pokemons():

    pokemons = read_in_pokemon("./data/gen9_pokemon.jsonl")

    # convert pokemons to just pokemon and associated stats in columns

    pokemon_stats = []

    for p in pokemons:
        pokemon_stats.append(
            {
                "name": str(p["name"]),
                "hp": int(extract_stat(p, "hp")),
                "attack": int(extract_stat(p, "attack")),
                "defense": int(extract_stat(p, "defense")),
                "specialattack": int(extract_stat(p, "special-attack")),
                "specialdefense": int(extract_stat(p, "special-defense")),
                "speed": int(extract_stat(p, "speed")),
            }
        )
    return pokemon_stats


# convert pokemon_stats to a sqlite database, and save it locally
# we will use this database to query the pokemon stats
# Your existing code


# Following code was generated with chatgpt to properly convert data to db
Base = declarative_base()


class Pokemon(Base):
    __tablename__ = "pokemon"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    hp = Column(Integer)
    attack = Column(Integer)
    defense = Column(Integer)
    specialattack = Column(Integer)
    specialdefense = Column(Integer)
    speed = Column(Integer)


def convert_to_sqlite(pokemon_stats, db_name="pokemon_stats.db"):

    # writes data to a sqlite database
    # written using chatgpt

    engine = create_engine(f"sqlite:///{db_name}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for pokemon in pokemon_stats:
        p = Pokemon(
            name=str(pokemon["name"]),
            hp=int(pokemon["hp"]),
            attack=int((pokemon["attack"])),
            defense=int(pokemon["defense"]),
            specialattack=int(pokemon["specialattack"]),
            specialdefense=int(pokemon["specialdefense"]),
            speed=int(pokemon["speed"]),
        )
        session.add(p)

    session.commit()
    session.close()


def create_chain():
    llm = Cohere(
        model="command", temperature=0, cohere_api_key=st.secrets["COHERE_API_KEY"]
    )
    # this should connect to the database in the data directory

    db = SQLDatabase.from_uri("sqlite:///pokemon_stats.db")
    # display intermediate results
    chain = SQLDatabaseChain.from_llm(llm, db)
    return chain


def run_query(chain, query):
    return chain.run(query)


def handle_bst_query(q):
    chain = create_chain()

    try:
        query_result = run_query(chain, q)
        return query_result

    except OperationalError:
        return "Query couldn't be transformed. Please try again by rephrasing."

    except ValueError or CohereAPIError:
        return "Input rejected. Please try again by rephrasing."


def handle_query(q):
    chain = create_chain()
    return run_query(chain, q)


def classify_intent(prompt):
    # given a string, classifies it
    co = cohere.Client(st.secrets["COHERE_API_KEY"])  # This is your trial API key
    response = co.classify(
        model="bfb1f19a-afaa-4faf-89db-f35df53f9de6-ft", inputs=[prompt]
    )

    return response.classifications[0].prediction


if __name__ == "__main__":
    # need some try except blocks to handle sql query failing
    # and Cohere generation errors if illicit input is passed

    # eventually, will integrate nemo guardails to prevent illicit input
    # for now, just try except blocks

    # if the pokemon db doesn't exist, create it
    parser = argparse.ArgumentParser(description="A chatbot for querying pokemon stats")
    parser.add_argument("ask", type=str, help="The query to run")

    args = parser.parse_args()
    ask = args.ask
    if not os.path.exists("./pokemon_stats.db"):
        pokemon_stats = preprocess_pokemons()
        convert_to_sqlite(pokemon_stats)
    print(handle_bst_query(ask))


# we need to convert my json file of pokemon stats into a sqlite database
