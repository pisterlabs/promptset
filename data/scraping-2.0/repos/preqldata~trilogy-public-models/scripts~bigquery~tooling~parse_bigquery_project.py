# requires openai
# and langchain

from typing import TYPE_CHECKING
from preql.core.models import (
    Datasource,
    ColumnAssignment,
    Environment,
    Concept,
    Metadata,
    Grain,
)
from preql.core.enums import DataType, Purpose
from preql.parsing.render import render_environment
import re
import os
from pathlib import Path
import json


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


if TYPE_CHECKING:
    from google.cloud import bigquery


def write_ds_file():
    pass


def get_table_keys(table: "bigquery.Table"):
    from langchain.llms import OpenAI

    llm = OpenAI(temperature=0.99, max_retries=1)
    columns = "\n".join([f"{c.name}:{c.description}" for c in table.schema])
    text = f"""Given a list of the following pairs of columns and descriptions for a SQL table, which column
or set of columns are the primary keys for the table?

Output the answer as a list of JSON array formatted column names with quotes around them.
Example responses:

- ["user_id", "order_id"]
- ["ssn"]
- ["customer_id"]
- ["date", "search_term"]

Columns are:
{columns}
Answer:
    """  # noqa: E501
    results = llm(text)
    print(results)
    return json.loads(results)


def process_description(input):
    if not input:
        return None
    return " ".join([x.strip() for x in input.split("\n")])


def parse_column(
    c: "bigquery.SchemaField", keys: list[str], parents: list | None = None
) -> list[Concept]:
    parents = []
    type_map = {
        "STRING": DataType.STRING,
        "INTEGER": DataType.INTEGER,
        "BOOLEAN": DataType.BOOL,
        "TIMESTAMP": DataType.TIMESTAMP,
        "FLOAT": DataType.FLOAT,
    }
    if c.field_type == "RECORD":
        output = []
        for x in c.fields:
            output.extend(parse_column(x, keys=keys, parents=parents + [c.name]))
        return output
    purpose = Purpose.KEY
    if c.name in keys:
        purpose = Purpose.KEY
    else:
        purpose = Purpose.PROPERTY

    return [
        Concept(
            name=camel_to_snake(c.name),
            metadata=Metadata(description=process_description(c.description)),
            datatype=type_map[c.field_type],
            purpose=purpose,
        )
    ]


def get_table_environment(table: "bigquery.Table", target: Path) -> Environment:
    snake = camel_to_snake(table.table_id)
    from preql.parser import parse

    fpath = target / (snake + ".preql")
    if not fpath.exists():
        return Environment(working_path=target)
    with open(fpath, "r", encoding="utf-8") as f:
        print(f"{fpath} already exists, returning existing environment")
        contents = f.read()
        env = Environment(working_path=target)
        environment, statements = parse(contents, environment=env)
        return environment


def process_table(table, client: "bigquery.Client", target: Path) -> Environment:
    environment = get_table_environment(table, target=target)
    # environment = Environment()
    columns = []
    grain = [c for c in environment.concepts.values() if c.purpose == Purpose.KEY]
    existing_bindings = set()
    # if there are already keys defined, defer to that
    # otherwise attempt to get keys from NLP
    keys = (
        [c.name for c in environment.concepts.values() if c.purpose == Purpose.KEY]
        or get_table_keys(table)
        or []
    )
    for _, datasource in environment.datasources.items():
        for c in datasource.columns:
            existing_bindings.add(c.alias)
    for c in table.schema:
        if c.name in existing_bindings:
            continue
        concepts = parse_column(c, keys=keys)
        if c.name in keys:
            grain.extend(concepts)
        for concept in concepts:
            environment.add_concept(concept, add_derived=False)
            assignment = ColumnAssignment(alias=c.name, concept=concept)
            columns.append(assignment)
    if not grain:
        raise ValueError(f"No grain found for table {table.table_id} keys {keys}")
    for concept in environment.concepts.values():
        if concept.purpose == Purpose.PROPERTY:
            concept.keys = grain
    datasource = environment.datasources.get(table.table_id)
    if datasource:
        for c in columns:
            datasource.columns.append(c)
    if not datasource:
        datasource = Datasource(
            columns=columns,
            identifier=table.table_id,
            address=table.full_table_id.replace(":", "."),
            grain=Grain(components=grain),
        )

    environment.datasources[table.table_id] = datasource
    return environment


def parse_public_bigquery_project(
    dataset: str, write: bool, project="bigquery-public-data"
):
    from google import auth
    from google.cloud import bigquery

    root = Path(__file__).parent.parent.parent
    target = Path(root) / "bigquery" / dataset
    cred, project = auth.default()
    client = bigquery.Client(credentials=cred, project=project)

    dataset_instance = client.get_dataset(
        dataset,
    )
    entrypoints = []
    for table_ref in client.list_tables(dataset=dataset_instance):
        table = client.get_table(table_ref)

        ds = process_table(table, client=client, target=target)
        snake = camel_to_snake(table.table_id)
        entrypoints.append(snake)
        if write:
            os.makedirs(target, exist_ok=True)
            path = target / (snake + ".preql")
            with open(path, "w") as f:
                f.write(render_environment(ds))
    if write:
        os.makedirs(target, exist_ok=True)
        init = """from trilogy_public_models.inventory import parse_initial_models

model = parse_initial_models(__file__)
"""
        path = target / "__init__.py"
        with open(path, "w") as f:
            f.write(init)
        entrypoint = target / "entrypoint.preql"
        with open(entrypoint, "w") as f:
            entrypoints = "\n".join([f"import {z} as {z};" for z in entrypoints])
            f.write(entrypoints)


if __name__ == "__main__":
    # ttl-test-355422.aoe2.match_player_actions
    parse_public_bigquery_project("aoe2", write=True, project="ttl-test-355422")
