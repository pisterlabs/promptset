import os
from langchain.vectorstores import SQLiteVSS
from _config import _config as _config_
from _common import _common as _common_
from _util import _util_directory as _util_directory_


@_common_.exception_handler
def get_vector_db(embedding, texts=None) -> SQLiteVSS:
    _config = _config_.PGConfigSingleton()
    _util_directory_.create_directory(os.path.dirname(_config.config.get("vector_db_default_location")))

    return SQLiteVSS.from_texts(
        texts=texts,
        embedding=embedding,
        table=_config.config.get("vector_db_default_table"),
        db_file=_config.config.get("vector_db_default_location")
)
