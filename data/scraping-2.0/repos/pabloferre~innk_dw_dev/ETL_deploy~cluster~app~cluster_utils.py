#Modulo de funciones generales para ETL
import os
import openai
import time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from dotenv import load_dotenv
import redshift_connector

def get_conn(aws_host, aws_db, aws_port, aws_user_db, aws_pass_db):
    """Get connection to Redshift database

    Args:
        aws_host (_type_): host
        aws_db (_type_): database/dw
        aws_port (_type_): port
        aws_user_db (_type_): user for database
        aws_pass_db (_type_): password for user

    Returns:
        conn: database connection
    """
    conn = redshift_connector.connect(
        host=aws_host,
        database=aws_db,
        port=aws_port,
        user=aws_user_db,
        password=aws_pass_db
    )
    return conn

def execute_sql(query:str, conn:redshift_connector):
    """Execute sql query in database

    Args:
        query (str): sql query
        conn (object): connection to database

    Returns:
        result: result of query
    """
    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchall()
    return result

def average_valid_vectors(row: pd.Series, columns: list):
    vectors = []
    expected_length = 1536
    for col in columns:
        if row[col] is not None and type(row[col]) != float:
            arr = np.array(row[col])
            if len(arr) != expected_length:
                continue
            vectors.append(arr)

    if not vectors:
        return np.nan

    stacked_vectors = np.vstack(vectors)
    mean_vector = np.nanmean(stacked_vectors, axis=0)

    return mean_vector


def deserialize_vector(vector_str:str)->np.ndarray:
    """Deserialize a vector from a string   

    Args:
        vector_str (str): string representation of a embedded vector

    Returns:
        np.ndarray: embeded vector
    """
    if vector_str == None:
        return np.nan
    return np.fromstring(vector_str, sep=',')