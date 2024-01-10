from langchain.sql_database import SQLDatabase
from urllib.parse import quote  
from sqlalchemy import MetaData, Table


class SQLDatabaseLangchainUtils:
    def __init__(self, db_connection, include_tables=[], samples=0, driver = 'oracle', schema=None):
        self.db_connection = db_connection
        self.include_tables = include_tables
        self.sample_rows_in_table_info = samples
        self.schema = schema
        if driver=='oracle':
            self.db = self.get_connection()
        elif driver=='mysql':
            self.db = self.get_connection_mysql()
        
        
    def get_connection(self):
        """Construct a SQLAlchemy engine from URI."""
        try:
            username = str(self.db_connection['DB_USER_NAME'])
            passwd = str(self.db_connection['DB_PASS'])
            hostname = str(self.db_connection['DB_HOST'])
            port = str(self.db_connection['DB_PORT'])
            database = str(self.db_connection['DB_NAME'])
            sqldriver = str(self.db_connection['SQL_DRIVER'])
            servicename = str(self.db_connection['SERVICE_NAME'])

            dsnStr = f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={hostname})(PORT={port}))(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME={servicename})))"
            uri = "{}://{}:{}@{}".format(sqldriver, username, quote(passwd), dsnStr)
                 
            if len(self.include_tables) > 0:       
                db = SQLDatabase.from_uri(uri, include_tables=self.include_tables, sample_rows_in_table_info=self.sample_rows_in_table_info, schema=self.schema)
            else:
                db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=self.sample_rows_in_table_info, schema=self.schema)
            return db
        except:
            raise ConnectionError("Error connecting to the database")
        
    def get_connection_mysql(self):
        """Construct a SQLAlchemy engine from URI."""
        try:
            username = str(self.db_connection['DB_USER_NAME'])
            passwd = str(self.db_connection['DB_PASS'])
            host = str(self.db_connection['DB_HOST'])
            port = str(self.db_connection['DB_PORT'])
            database = str(self.db_connection['DB_NAME'])
            

            uri = f"mysql://{username}:{passwd}@{host}:{port}/{database}"
        

                        
            if len(self.include_tables) > 0:       
                db = SQLDatabase.from_uri(uri, include_tables=self.include_tables, sample_rows_in_table_info=self.sample_rows_in_table_info)
            else:
                db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=self.sample_rows_in_table_info)
            return db
        except:
            raise ConnectionError("Error connecting to the database")

    def run(self, query: str):
        try:
            """Execute a SQL command and return a string representing the results."""
            response = self.db.run(query)
            return response
        except Exception as e:
            return f"Error: {e}"    
        
    def get_table_info(self, tables = []):
        """
            Get information about specified tables.
        """
        if len(tables) > 0:
            return self.db.get_table_info(tables)
        return self.db.get_table_info()
    
    def get_table_names(self):
        """
            Get names of tables available.
        """
        return self.db.get_usable_table_names()
    
    def get_dialect(self):
        """
            Get dialect of the database.
        """
        return self.db.dialect
    
    def check_table_exist(self, table_name):
        return table_name in self.get_usable_table_names()
    
    def check_column_exist(self, column_name, table_name):
        table_schema = self.get_table_info([table_name])
        return column_name in table_schema
    
    def get_metadata(self):
        return self.db._metadata.sorted_tables
    
    def get_primary_keys(self, tables = []):
        primary_keys = []
        for tbl in self.get_metadata():
            if len(tables)==0 or tbl.name in tables:
                for column in tbl.columns.items():
                    column = column[1]
                    if column.primary_key:
                        primary_keys.append(f"{tbl.name}.{column.name}")
        return primary_keys

    def get_foreign_keys(self, tables = []):
        foreign_keys = []
        for tbl in self.get_metadata():
            if len(tables)==0 or tbl.name in tables:
                for column in tbl.columns.items():
                    fks = list(column[1].foreign_keys)
                    if len(fks) > 0:
                        column_name = f"{tbl.name}.{column[0]}"
                        fk_column = list(column[1].foreign_keys)[0].column
                        foreign_keys.append(f"{column_name}={fk_column}")
        return foreign_keys
    
    def get_schema_json(self, tables = []):
        schema_json = {}
        for tbl in self.get_metadata():
            if len(tables)==0 or tbl.name in tables:
                schema_json[tbl.name] = [c.name for c in tbl._columns]
        return schema_json
    
    
    def get_schema_openai_prompt(self, tables = []):
        
        schema_pompt = ""
        for tbl in self.get_metadata():
            if len(tables)==0 or tbl.name in tables:
                schema_pompt += f"# {tbl.name} ({', '.join([c.name for c in tbl._columns])})\n"
        return schema_pompt
    
    def get_primary_keys_openai_prompt(self, tables = []):
        primary_keys = self.get_primary_keys(tables)
        primary_keys_prompt = ""
        for pk in primary_keys:
            primary_keys_prompt += f"# {pk}\n"
        return primary_keys_prompt
    
    def get_foreign_keys_openai_prompt(self, tables = []):
        foreign_keys = self.get_foreign_keys(tables)
        foreign_keys_prompt = ""
        for fk in foreign_keys:
            foreign_keys_prompt += f"# {fk}\n"
        return foreign_keys_prompt
    
    def get_schema_basic_prompt(self, tables = []):
            
        schema_pompt = ""
        for tbl in self.get_metadata():
            if len(tables)==0 or tbl.name in tables:
                schema_pompt += f"Table {tbl.name}, columns = [*,{','.join([c.name for c in tbl._columns])}]\n"
        return schema_pompt
    
            
    def get_primary_keys_basic_prompt(self, tables = []):
        primary_keys = self.get_primary_keys(tables)
        primary_keys_prompt = f"[{','.join([pk for pk in primary_keys])}]\n"
        return primary_keys_prompt
    
    def get_foreign_keys_basic_prompt(self, tables = []):
        foreign_keys = self.get_foreign_keys(tables)
        foreign_keys_prompt = f"[{','.join([fk for fk in foreign_keys])}]\n"
        return foreign_keys_prompt
    

    
    
    


