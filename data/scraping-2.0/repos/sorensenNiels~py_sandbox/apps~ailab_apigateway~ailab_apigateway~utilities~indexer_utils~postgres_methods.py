from langchain.vectorstores.pgvector import PGVector

from ...config import settings


def make_connection_string() -> str:
    """Function taking a dictionary containing informations on a postgres database
    and formats it into a connection string
    """

    connection_string = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=settings.database_host,
        port=int(settings.database_port),
        database=settings.database_name,
        user=settings.database_username,
        password=settings.database_password,
    )
    return connection_string
