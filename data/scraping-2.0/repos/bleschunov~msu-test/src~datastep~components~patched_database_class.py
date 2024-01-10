from langchain import SQLDatabase
from langchain.utilities.sql_database import text


class SQLDatabasePatched(SQLDatabase):
    def run(self, command: str, fetch: str = "all") -> list:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                if self.dialect == "snowflake":
                    connection.exec_driver_sql(
                        f"ALTER SESSION SET search_path='{self._schema}'"
                    )
                elif self.dialect == "bigquery":
                    connection.exec_driver_sql(f"SET @@dataset_id='{self._schema}'")
                else:
                    connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                # if fetch == "all":
                #    result = cursor.fetchall()
                # elif fetch == "one":
                #    result = cursor.fetchone()  # type: ignore
                # else:
                #    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                result = cursor.mappings().all()  # PATCHED

                # Convert columns values to string to avoid issues with sqlalchmey
                # trunacating text
                # if isinstance(result, list):
                #    return [
                #        tuple(
                #            truncate_word(c, length=self._max_string_length) for c in r
                #        )
                #        for r in result
                #    ]
                #
                # return tuple(
                #    truncate_word(c, length=self._max_string_length) for c in result
                # )
                return result
        return []
