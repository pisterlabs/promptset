import os
import openai
import mysql.connector
from openai import OpenAI
from prettytable import PrettyTable
from database_utils import DatabaseUtils
from analyze_database import AnalyzeDatabase

class SQLInfo:
    # Initialize the SQLInfo class with two database cursors and command line arguments
    def __init__(self, cursor1, cursor2, args):
        # Create an OpenAI client using the API key from the environment variables
        self.client = OpenAI ( api_key=os.environ.get('OPENAI_API_KEY'),)
        self.cursor1 = cursor1
        self.cursor2 = cursor2
        self.args = args
    
    # Check if the OpenAI API key is set in the environment variables
    def check_for_sql_key(self):
        if os.environ.get('OPENAI_API_KEY') is None:
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("""Please set the OPENAI_API_KEY environment variable to your OpenAI API key.
                  If you do not have an OpenAI API key, please visit https://platform.openai.com/ to get one.""")
            return False
        else:
            return True
        
    # Use OpenAI to generate a SQL query based on the user's input
    def openai_sql_query(self, user_input):
        
        try:
            prompt = f"Generate a SQL query to {user_input}. Only print the query"
            # Send a chat completion request to the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            # Return the content of the first choice in the response
            return response.choices[0].message.content.strip()
        
        # Handle various types of errors that can occur when calling the OpenAI API
        except openai.RateLimitError as e:
            print(f"OpenAI API rate limit exceeded: {e}")
        except openai.AuthenticationError as e:
            print(f"OpenAI API authentication error: {e}")
        except openai.InvalidRequestError as e:
            print(f"OpenAI API invalid request error: {e}")
        except openai.APIError as e:
            print(f"OpenAI API returned API Error: {e}")
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")

    # Execute a SQL query on one or two databases
    def execute_sql_query(self, sql_query):
        if sql_query is None:
            print("No SQL query provided.")
            return

        # If the query is a SELECT statement, execute it and print the results
        if sql_query.strip().upper().startswith("SELECT"):
            if self.cursor2 is None:
                self.cursor1.execute(sql_query)
                print(self.cursor1.fetchall())
                return
            else:
                self.cursor1.execute(sql_query)
                print(self.cursor1.fetchall())
                self.cursor2.execute(sql_query)
                print(self.cursor2.fetchall())
                return
        # If the query is not a SELECT statement, just execute it
        else:
            if self.cursor2 is None:
                self.cursor1.execute(sql_query)
                return
            else:
                self.cursor1.execute(sql_query)
                self.cursor2.execute(sql_query)
                return
            
    # Get the user's input, generate a SQL query using OpenAI, and execute the query
    def get_input(self):
        user_input = input("What information would you like to query?")
        sql_query = self.openai_sql_query(user_input)
        print(sql_query)
        self.execute_sql_query(sql_query)

    # List the databases on a server
    def list_databases(self, server, user, password, host, port):
        # Connect to the server
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        cursor = conn.cursor()
        # Get the names of the databases on the server
        databases = DatabaseUtils.get_database_names(cursor)

        # Create a PrettyTable instance for the databases
        table = PrettyTable()
        table.field_names = ["Databases on {}".format(server)]
        table.align["Databases on {}".format(server)] = 'l'
        for database in databases:
            table.add_row([database])

        conn.close()
        return table  # Return the table instead of printing it
    
    def list_tables(self, user, password, host, port, database):
        # Connect to the server
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )

        cursor = conn.cursor()

        tables = DatabaseUtils.get_table_names(cursor, database)

        # Print the names of the databases in a table
        output_table = PrettyTable(["{} Tables in {}".format(len(tables),database)])
        output_table.align["{} Tables in {}".format(len(tables), database)] = 'l'
        for table in tables:
            output_table.add_row([table])

        conn.close()
        
        return output_table

    def main(self):

        if self.args['list']:
            user1, password1, host1, port1 = DatabaseUtils.parse_connection_string(self.args['server1'])
            table1 = str(self.list_databases("Server 1", user1, password1, host1, port1)).splitlines()
            
            if self.args['server2']:
                user2, password2, host2, port2 = DatabaseUtils.parse_connection_string(self.args['server2'])
                table2 = str(self.list_databases("Server 2", user2, password2, host2, port2)).splitlines()
                
                # Get the maximum number of lines in the two tables
                max_lines = max(len(table1), len(table2))
                
                # Print the tables side by side
                for i in range(max_lines):
                    row1 = table1[i] if i < len(table1) else ''
                    row2 = table2[i] if i < len(table2) else ''
                    print(f"{row1:<30} {row2}")
            else:
                print('\n'.join(table1))  # Print only the first table

        if self.args['analyze']:
            if self.args['database'] is None:
                print("Error: -db, --database is required. with -A, --analyze")
                return
            else:
                AnalyzeDatabase(self.cursor1, self.cursor2, self.args).main()
        if self.args['tables']:
            if self.args['database'] is None:
                print("Error: -db, --database is required. with -T, --tables")
                return
            else:
                user1, password1, host1, port1 = DatabaseUtils.parse_connection_string(self.args['server1'])
                table1 = str(self.list_tables(user1, password1, host1, port1, self.args['database'])).splitlines()
                if self.args["server2"]:
                    user2, password2, host2, port2 = DatabaseUtils.parse_connection_string(self.args['server2'])
                    table2 = str(self.list_tables(user2, password2, host2, port2, self.args['database'])).splitlines()
                                    # Get the maximum number of lines in the two tables
                    max_lines = max(len(table1), len(table2))
                    
                    # Print the tables side by side
                    for i in range(max_lines):
                        row1 = table1[i] if i < len(table1) else ''
                        row2 = table2[i] if i < len(table2) else ''
                        print(f"{row1:<30} {row2}")
                else:
                    print('\n'.join(table1))  # Print only the first table

        if self.args['sql_query'] is not None or self.args['openai']:
            if self.args['sql_query'] is not None:
                if self.args['sql_query'] == "" and not self.args['openai']:
                    print("Error: Must be passed an SQL query or used with --openai option.")
                    return
                else:
                    self.execute_sql_query(self.args['sql_query'])
            elif self.args['openai'] and self.args['sql_query'] is None:
                print("Error: --openai must be used with --sql-query.")
                return
            else:
                print("Error: Your SQL Query or --openai is required.")
                return