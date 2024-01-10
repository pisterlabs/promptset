import os
import time
import json
import openai
import weaviate
import pandas as pd
import weaviate.classes as wvc
from config import ROLE_MESSAGE, WILDCARD


def format_input(item: str, restrictions: list[str]) -> str:
    formatted = f"""
    item: {item}
    categories:
    """
    for i, restriction in enumerate(restrictions):
        formatted += f"{i+1}. {restriction}\n"
    return formatted


def get_weaviate_client() -> weaviate.Client:
    """
    Connect to a local Weaviate instance deployed using Docker compose with standard port configurations.

    Parameters:
        None
    Returns
        `weaviate.WeaviateClient`
            The client connected to the local instance
    """
    client = weaviate.connect_to_local(
        port=8080,
        grpc_port=50051,
        headers={
            "X-OpenAI-API-Key": os.getenv("OPENAI_API_KEY"),
        },
    )

    return client


def get_openai_client() -> openai.OpenAI:
    return openai.Client()


def embed(
    weaviate_client: weaviate.Client,
    path_to_restriction_data: str = "data.csv",
) -> None:
    """
    Vectorize and embed restriction data into weaviate

    Parameters:
        client: weaviate.Client
            The client connected to the local instance
        path_to_restriction_data: str
            The path to the restriction data csv file.
    Returns
        None
    """
    # Define class
    weaviate_client.collections.delete("Restriction")
    restrictions = weaviate_client.collections.create(
        name="Restriction",
        vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(),
        properties=[
            wvc.Property(
                name="hs_code",
                data_type=wvc.DataType.TEXT,
                skip_vectorization=True,
            ),
            wvc.Property(name="item", data_type=wvc.DataType.TEXT),
            wvc.Property(
                name="restriction_text",
                data_type=wvc.DataType.TEXT,
                skip_vectorization=True,
            ),
        ],
    )

    # Read in data
    interesting_columns = [
        "hs_code",
        "item",
        "restriction",
    ]
    df = pd.read_csv(path_to_restriction_data)[interesting_columns]
    data = df.to_dict(orient="records")
    restriction_objs = list()

    for d in data:
        restriction_objs.append(
            {
                "hs_code": d["hs_code"],
                "item": d["item"],
                "restriction_text": d["restriction"],
            }
        )

    restrictions = weaviate_client.collections.get("Restriction")
    restrictions.data.insert_many(restriction_objs)

    return


def get_filters(code: str) -> wvc.Filter:
    """
    Creates filters in order to get only restrictions with applicable hs codes.
    Example: if the code is 0207, then 0, 02, 020, 0207, and anything that starts with 0207 apply,
            Any restrictions that apply to all hs_codes, represented by WILDCARD, also apply.

    Parameters:
        code: str
            The hs code to create filters for
    Returns
        wvc.Filter
            The filters to apply to the query
    """
    filters = wvc.Filter(path="hs_code").like(f"{code}*") | wvc.Filter(
        path="hs_code"
    ).equal(
        WILDCARD
    )  # TODO ENABLE WILDCARD
    built_code = ""
    for c in code:
        built_code += c
        filters = filters | wvc.Filter(path="hs_code").equal(built_code)
    return filters


def get_neighbors(
    item_description: str,
    item_hs_code: str,
    weaviate_client: weaviate.Client,
    debug: bool = False,
) -> list:
    """
    Returns all of the 'neighbors' of a given item description, filtering on the hs code.

    Parameters:
        item_description: str
            The item description to search for
        item_hs_code: str
            The hs code to filter on
        client: weaviate.Client
            The client connected to the local instance
        debug: bool
            Whether or not to print debug statements
    Returns
        reponse.objects: list
            A list of weaviate objects that are neighbors to the given item description
    """
    restrictions = weaviate_client.collections.get("Restriction")

    response = restrictions.query.near_text(
        query=item_description,
        filters=get_filters(item_hs_code),
        return_metadata=wvc.MetadataQuery(distance=True),
    )

    if len(response.objects) == 0:
        return []

    if debug:
        print("---")
        print(f"FILTERING ON '{item_hs_code}'")
        for object in response.objects:
            print(f"""IN:({object.properties["hs_code"]})""")
        print("---")

        print("CLOSEST NEIGHBOR:")
        print(response.objects[0].properties)
        print("---")

    return response.objects


def get_openai_response(input) -> str:
    """
    Gets a response from OpenAI.

    Parameters:
        input: str
    Returns
        response: str
            The response from OpenAI
    """
    return openai_client.chat.completions.with_raw_response.create(
        messages=[
            {
                "role": "system",
                "content": ROLE_MESSAGE,
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        model="gpt-3.5-turbo",
    )


def process_row(
    row, weaviate_client: weaviate.Client, openai_client: openai.OpenAI
) -> list:
    """
    Gets the neighbors for a row and returns their data formatted as a list of dictionaries.

    Parameters:
        row: pd.Series
            The row to process
        client: weaviate.Client
            The client connected to the local instance
    Returns
        new_rows: list
            A list of dictionaries containing the data for the neighbors of the row
    """
    t1 = time.time()
    response_objects = get_neighbors(
        row["description"], row["hs_code"], weaviate_client=weaviate_client
    )
    time_to_get_neighbors = time.time() - t1
    new_rows = []
    restricted_items = []
    for object in response_objects:
        o = object.properties
        # Create restricted item list
        restricted_items.append(o["item"])

    if len(restricted_items) == 0:
        # No restrictions
        new_rows.append(
            {
                "hs_code": row["hs_code"],
                "description": row["description"],
                "restricted_codes": "",
                "restricted_item": "",
                "restriction": "",
                "distance": 0,
                "time_to_get_neighbors": time_to_get_neighbors,
                "time_to_get_response": "NA",
                "time_to_process_row": time.time() - t1,
                "prompt_tokens": "NA",
                "completion_tokens": "NA",
                "total_tokens": "NA",
            }
        )
        return new_rows, time_to_get_neighbors, "NA", time.time() - t1, "NA", "NA", "NA"

    # Create input
    input = format_input(row["description"], restricted_items)

    # Send to OpenAI
    t2 = time.time()
    response = get_openai_response(input)
    time_to_get_response = time.time() - t2
    response = json.loads(response.text)
    prompt_tokens = response["usage"]["prompt_tokens"]
    completion_tokens = response["usage"]["completion_tokens"]
    total_tokens = response["usage"]["total_tokens"]
    # Parse response
    if response["choices"][0]["finish_reason"] != "stop":
        raise Exception("OpenAI did not finish processing the prompt.")
    choices = (
        response["choices"][0]["message"]["content"]
        .replace("\n", "")
        .replace(" ", "")
        .split(",")
    )

    for choice in choices:
        choice = choice.replace(".", "")
        if choice == "" or choice == "0":
            continue
        try:
            choice = int(choice)
        except ValueError:
            if (
                "norestrictionsapply" in choice.lower()
                or "doesnotapply" in choice.lower()
                or "nocategoryapplies" in choice.lower()
            ):
                new_rows.append(
                    {
                        "hs_code": row["hs_code"],
                        "description": row["description"],
                        "restricted_codes": "",
                        "restricted_item": "",
                        "restriction": "",
                        "distance": 0,
                        "time_to_get_neighbors": time_to_get_neighbors,
                        "time_to_get_response": time_to_get_response,
                        "time_to_process_row": time.time() - t1,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                )
                return (
                    new_rows,
                    time_to_get_neighbors,
                    time_to_get_response,
                    time.time() - t1,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
            else:
                raise Exception("OpenAI returned a non-integer choice.")
        if choice > len(restricted_items):
            raise Exception("OpenAI returned a choice that is out of bounds.")
        # get associated object from response_objects
        o = response_objects[choice - 1].properties
        new_rows.append(
            {
                "hs_code": row["hs_code"],
                "description": row["description"],
                "restricted_codes": o["hs_code"],
                "restricted_item": o["item"],
                "restriction": o["restriction_text"],
                "distance": response_objects[choice - 1].metadata.distance,
                "time_to_get_neighbors": time_to_get_neighbors,
                "time_to_get_response": time_to_get_response,
                "time_to_process_row": time.time() - t1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )

    time_to_process_row = time.time() - t1
    return (
        new_rows,
        time_to_get_neighbors,
        time_to_get_response,
        time_to_process_row,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def restrict_from_csv(
    weaviate_client: weaviate.Client,
    openai_client: openai.OpenAI,
    filepath: str,
    output_path: str,
    encoding: str = "utf8",
) -> pd.DataFrame:
    """
    Gets all neighbors for each item in a csv and formats them into a dataframe.

    Parameters:
        client: weaviate.Client
            The client connected to the local instance
        filepath: str
            The path to the csv file
        encoding: str
            The encoding of the csv file
    Returns
        new_rows: list
            A list of dictionaries containing the data for the neighbors of the row
    """
    queries = pd.read_csv(filepath, encoding=encoding)
    queries["hs_code"] = queries["hs_code"].astype(str)
    queries["hs_code"] = queries["hs_code"].str.replace(".", "")

    # Times & Tokens
    time_to_get_neighbors_list = []
    time_to_get_response_list = []
    time_to_process_row_list = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []

    columns = [
        "hs_code",
        "description",
        "restricted_codes",
        "restricted_item",
        "restriction",
        "distance",
        "time_to_get_neighbors",
        "time_to_get_response",
        "time_to_process_row",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    new_rows = []
    for index, row in queries.iterrows():
        if index % 10 == 0:
            print(f"{index}/{len(queries)} rows processed.")
        (
            processed_row,
            time_to_get_neighbors,
            time_to_get_response,
            time_to_process_row,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ) = process_row(
            row, weaviate_client=weaviate_client, openai_client=openai_client
        )
        new_rows.extend(processed_row)
        if time_to_get_neighbors != "NA":
            time_to_get_neighbors_list.append(time_to_get_neighbors)
        if time_to_get_response != "NA":
            time_to_get_response_list.append(time_to_get_response)
        if time_to_process_row != "NA":
            time_to_process_row_list.append(time_to_process_row)
        if prompt_tokens != "NA":
            prompt_tokens_list.append(prompt_tokens)
        if completion_tokens != "NA":
            completion_tokens_list.append(completion_tokens)
        if total_tokens != "NA":
            total_tokens_list.append(total_tokens)

    print(f"Finished processing {len(queries)} rows.")
    print(
        f"Average time to get neighbors: {sum(time_to_get_neighbors_list)/len(time_to_get_neighbors_list)}"
    )
    print(
        f"Average time to get response: {sum(time_to_get_response_list)/len(time_to_get_response_list)}"
    )
    print(
        f"Average time to process row: {sum(time_to_process_row_list)/len(time_to_process_row_list)}"
    )
    print(f"Average prompt tokens: {sum(prompt_tokens_list)/len(prompt_tokens_list)}")
    print(
        f"Average completion tokens: {sum(completion_tokens_list)/len(completion_tokens_list)}"
    )
    print(f"Average total tokens: {sum(total_tokens_list)/len(total_tokens_list)}")

    rdf = pd.DataFrame(new_rows, columns=columns)
    rdf.to_csv(output_path, index=False)
    return rdf


if __name__ == "__main__":
    weaviate_client = get_weaviate_client()
    openai_client = get_openai_client()
    # embed(
    #     weaviate_client=weaviate_client,
    #     path_to_restriction_data="data/canada_restrictions.csv",
    # )
    # get_neighbors("chicken", "0207", debug=True)

    restrict_from_csv(
        filepath="data/walmart_input.csv",
        output_path="data/walmart_output.csv",
        encoding="latin1",
        weaviate_client=weaviate_client,
        openai_client=openai_client,
    )
