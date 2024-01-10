#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function to geocode incidents that are not currently
in the geocoded table, based on ICRs passed from
new records via the scraping function, into the database.

@Author: Luke Zaruba
@Date: Aug 9, 2023
@Version: 0.0.0
"""

import os

import functions_framework
import openai
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# AI Standardization
def _generate_ai_response(input_text):
    openai.api_key = os.environ["OPENAI_KEY"]

    prompt = "Standardize the following location description into text that could be fed into a Geocoding API. When responding, only return the output text."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ],
        temperature=0.7,
        n=1,
        max_tokens=150,
        stop=None,
    )

    return response.choices[0].message.content.strip().split("\n")[-1]


# Geocoding
def _geocode_address(address):
    base_url = "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {
        "f": "json",
        "singleLine": address,
        "maxLocations": "1",
        "token": os.environ["GEOCODE_TOKEN"],
    }

    response = requests.get(base_url, params=params)

    data = response.json()

    return (
        data["candidates"][0]["location"]["x"],
        data["candidates"][0]["location"]["y"],
    )


# Get Existing Data for ICRs - if passed in, use those, else, compare raw and geo and use missing icrs not in geo
def extract_existing_data(db, icr_tuple=None):
    if icr_tuple is None:
        raw_query = "SELECT icr FROM raw_accidents"
        geo_query = "SELECT icr FROM geo_accidents"

        # Execute the query and fetch all results
        with db.connect() as connection:
            result_raw = connection.execute(text(raw_query))
            rows_raw = result_raw.fetchall()

            existing_icr_raw = [row[0] for row in rows_raw]

            result_geo = connection.execute(text(geo_query))
            rows_geo = result_geo.fetchall()

            existing_icr_geo = [row[0] for row in rows_geo]

        icr_tuple = tuple(
            [icr for icr in existing_icr_raw if icr not in existing_icr_geo]
        )

    # Get Existing Data from DB
    extract_query = f"SELECT * FROM raw_accidents WHERE icr IN {tuple(icr_tuple)}"

    with db.connect() as connection:
        extract_result = connection.execute(text(extract_query))
        extract_rows = extract_result.fetchall()

    return extract_rows


def geocode(rows):
    upload_list = []

    for record in rows:
        try:
            # Standardization
            # ai_result = _generate_ai_response(record[4])
            # new_record = tuple(record) + (ai_result,)

            new_record = tuple(record)

            # Geocoding
            gc_result = _geocode_address(new_record[4])  # 7 -> 4

            new_record += gc_result

            # Point WKT
            new_record += (f"POINT({new_record[7]} {new_record[8]})",)  # 8 -> 7, 9 -> 8

            # Fix Times
            new_record = (
                new_record[:2]
                + (new_record[2].strftime("%Y-%m-%d %H:%M:%S"),)
                + (new_record[3],)
                + (new_record[4].replace("'", "''"),)
                + new_record[5:]
            )

            upload_list.append(new_record)
        except:
            pass

    return upload_list


def insert_records(db, insert_list):
    if len(insert_list) != 0:
        insert_query = "INSERT INTO geo_accidents (icr, incident_type, incident_date, district, location_description, road_condition, vehicles_involved, x, y, geom) VALUES "

        for record in insert_list:
            insert_query += f"{record}, "

        insert_query = insert_query[:-2]

        insert_query = insert_query.replace('"', "'")

        with db.connect() as connection:
            connection.execute(text(insert_query))
            connection.commit()
    else:
        return


@functions_framework.http
def main(manual=False):
    # Creating Database Engine Instance
    _db_url = URL.create(
        drivername="postgresql",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        database=os.environ["DB_NAME"],
    )
    db = create_engine(_db_url)

    # For Debugging
    if manual:
        manual_icr = tuple([23270768, 23200808])

        rows_to_gc = extract_existing_data(db, manual_icr)

        gc_out = geocode(rows_to_gc)

        insert_records(db, gc_out)

    else:
        rows_to_gc = extract_existing_data(db)

        gc_out = geocode(rows_to_gc)

        insert_records(db, gc_out)


if __name__ == "__main__":
    main()
