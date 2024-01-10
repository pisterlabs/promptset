from langchain.utilities import SQLDatabase

# from util.logger import log


class DatastepSqlDatabase:
    # @log("Подключение к базе данных")
    def __init__(
        self,
        database_connection_string: str,
        include_tables: list[str],
    ):
        # TODO: Использовать пулинг из Алхимии
        self.database = SQLDatabase.from_uri(
            database_connection_string,
            include_tables=include_tables,
            view_support=True
        )

    # @log("Исполнение SQL в базе данных")
    def run(self, sql_query):
        response = self.database._execute(sql_query)
        return response
