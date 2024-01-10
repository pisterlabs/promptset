from pathlib import Path
from langchain.utilities import SQLDatabase


DATABASE = SQLDatabase.from_uri(
    f"duckdb:///{Path(__file__).absolute().parent.parent}/data/duck_sql.db",
    sample_rows_in_table_info=1
)
