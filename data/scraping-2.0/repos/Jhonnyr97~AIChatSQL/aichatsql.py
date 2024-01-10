from langchain.utilities import SQLDatabase
from cat.mad_hatter.decorators import tool, hook
from langchain_experimental.sql import SQLDatabaseChain
import subprocess

@tool
def database(tool_input, cat):
    """This plugin needs tool_input (human message) to return the result from the database data in human natural
    language"""

    db = connect(cat)
    db_chain = SQLDatabaseChain.from_llm(cat._llm, db, verbose=True)

    return str(db_chain.run(tool_input))


@hook(priority=0)
def before_cat_bootstrap(cat) -> None:
    check_pkg_config();
    check_libmysqlclient();
    check_mysqlclient_module();


def check_pkg_config():
    subprocess.check_call(["apt-get", "-y", "update"])
    subprocess.check_call(["apt-get", "-y", "install", "pkg-config"])


def check_libmysqlclient():
    try:
        subprocess.check_call(["pkg-config", "--exists", "default-libmysqlclient"])
    except subprocess.CalledProcessError:
        print("Installing default-libmysqlclient-dev")
        subprocess.check_call(["apt-get", "-y", "install", "default-libmysqlclient-dev"])


def check_mysqlclient_module():
    try:
        import mysqlclient
    except ImportError:
        print("Installing mysqlclient")
        subprocess.check_call(["pip", "install", "mysqlclient"])


def connect(cat):
    settings = cat.mad_hatter.plugins["aichatsql"].load_settings()
    if settings["data_source"] == "sqlite":
        uri = f"sqlite:///cat/plugins/sqlite_db/{settings['host']}"
    elif settings["data_source"] == "postgresql":
        uri = f"postgresql+psycopg2://{settings['username']}:{settings['password']}@{settings['host']}:{settings['port']}/{settings['database']}"
    else:
        uri = f"mysql://{settings['username']}:{settings['password']}@{settings['host']}:{settings['port']}/{settings['database']}"

    return SQLDatabase.from_uri(uri,
                                include_tables=settings["allowed_tables"].split(", "),
                                sample_rows_in_table_info=2
                                )
