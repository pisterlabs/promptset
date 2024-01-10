from langchain.sql_database import SQLDatabase
from dotenv import load_dotenv
import os
import json
import pyodbc
class SQLDB:
    def __init__(self):
        load_dotenv()
        self.db_username = os.getenv("SQLDB_USERNAME")
        self.db_password = os.getenv("SQLDB_PASSWORD")
        self.db_host = os.getenv("SQLDB_HOST")
        self.db_name = os.getenv("SQLDB_NAME")
        self.db_include_tables = os.getenv("SQLDB_INCLUDE_TABLE")
        self.conn = pyodbc.connect(f"DRIVER=ODBC Driver 17 for SQL Server;Server={self.db_host};Database={self.db_name};UID={self.db_username};PWD={self.db_password}", autocommit=True)

    def config_sqldb(self):
        db_uri = f"mssql+pyodbc://{self.db_username}:{self.db_password}@{self.db_host}/{self.db_name}?driver=ODBC+Driver+17+for+SQL+Server"
        sql_db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=1, include_tables=json.loads(self.db_include_tables))
        return sql_db
    
    def autogenerate_ddl(self,table):
        cursor = self.conn.cursor()
        constraint = self.get_table_constraint(table)
        data = []
        for row in cursor.columns(table=table):
            data.extend([row])
            sql_template = "CREATE TABLE {table} (\n{sql_columns} {constraint})"
            columns = []
            for entry in data:
                column_definition = "{} {}({})".format(entry[3], entry[5], entry[7])
                columns.append(column_definition)

            sql_columns = ",\n".join(columns)
            sql_statement = sql_template.format(table=table,sql_columns=sql_columns,constraint=constraint)
        return sql_statement
    
    def get_table(self):
        cursor = self.conn.cursor()
        data = []
        for row in cursor.tables():
            obj = {}
            obj['value'] = row.table_name
            obj['text'] = row.table_name
            data.append(obj)
        return data

    def get_table_constraint(self,table_name):
        query = f'''
            SELECT 
                'CONSTRAINT [' + C.name + '] PRIMARY KEY(' + K.name + ')' AS PrimaryKey
            FROM 
                sys.indexes AS I
            INNER JOIN 
                sys.index_columns AS IC ON I.object_id = IC.object_id AND I.index_id = IC.index_id
            INNER JOIN 
                sys.columns AS K ON IC.column_id = K.column_id AND IC.object_id = K.object_id
            INNER JOIN 
                sys.tables AS T ON I.object_id = T.object_id
            INNER JOIN 
                sys.key_constraints AS C ON I.object_id = C.parent_object_id AND I.index_id = C.unique_index_id
            WHERE 
                T.name = '{table_name}' AND C.type = 'PK'
            UNION ALL
            SELECT 
                'CONSTRAINT [' + FK.name + '] FOREIGN KEY(' + CP.name + ') REFERENCES [' + RT.name + '] (' + RC.name + ')' AS ForeignKey
            FROM 
                sys.foreign_keys AS FK
            INNER JOIN 
                sys.tables AS TP ON FK.parent_object_id = TP.object_id
            INNER JOIN 
                sys.tables AS RT ON FK.referenced_object_id = RT.object_id
            INNER JOIN 
                sys.foreign_key_columns AS FKC ON FKC.constraint_object_id = FK.object_id
            INNER JOIN 
                sys.columns AS CP ON FKC.parent_column_id = CP.column_id AND FKC.parent_object_id = CP.object_id
            INNER JOIN 
                sys.columns AS RC ON FKC.referenced_column_id = RC.column_id AND FKC.referenced_object_id = RC.object_id
            WHERE 
                TP.name = '{table_name}'
        '''
        cursor = self.conn.cursor()
        cursor.execute(query)
        string = ''
        for row in cursor.fetchall():
            fg = str(row)
            string += '\n'+fg[2:-3]+','
        return string