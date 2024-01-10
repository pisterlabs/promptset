#!/usr/bin/env python3

import cmd
import argparse
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers import SqlLexer
from influxdb_client_3 import InfluxDBClient3
from .helper import config_helper
from .openai_helper import OpenAIHelper
from .file_writer import FileWriter

_usage_string = """
to write data use influxdb line protocol:
> influx3 write testmes,tag1=tagvalue field1=0.0 <optional timestamp>

to read data with sql:
> influx3 sql select * from testmes where time > now() - interval'1 minute'

to enter interactive mode:
> influx3
"""

_description_string = 'CLI application for Querying IOx with arguments and interactive mode.'

class IOXCLI(cmd.Cmd):
    intro = 'InfluxDB 3.0 CLI.\n'
    prompt = '(>) '

    def __init__(self):
        super(IOXCLI, self).__init__()
        self.config_helper = config_helper()
        self.active_config = self.config_helper._get_active()
        self._setup_client()
        self._sql_prompt_session = PromptSession(lexer=PygmentsLexer(SqlLexer))
        self._write_prompt_session = PromptSession(lexer=None)
       

    

    def do_query(self, arg, language):
        if self.active_config == {}:
            print("can't query, no active configs")
            return
        
        # Retrieve the iterator
        reader = self._query_chunks(arg, language)

        if reader is None:
            return
        
                # Create custom key bindings
        bindings = KeyBindings()

        # Flag to determine if there is more data
        has_more_data = True

        # Bind the 'tab' key
        @bindings.add('tab')
        def _(event):
            nonlocal has_more_data
            if has_more_data:
                try:
                    batch, buff = reader.read_chunk()
                    print(batch.to_pandas().to_markdown())
                except StopIteration:
                    print("End of data. Press Enter to continue.")
                    has_more_data = False
                except Exception as e:
                    print(e)
                    has_more_data = False

        # Bind the 'f' key
        @bindings.add('f')
        def _(event):
            nonlocal batch
            try:
                # Prompt the user for a file name
                file_name = input("Enter the file name with full path (e.g. /home/user/sample.json): ")
                # Save the current batch of data to the specified file
                wf = FileWriter()
                wf.write_file(name=file_name, d=batch)
                print(f"Data saved to {file_name}.")
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")

        # Create a session with the bindings
        session = PromptSession(key_bindings=bindings)
        try:
            batch, buff = reader.read_chunk()
            print(batch.to_pandas().to_markdown())
        except StopIteration:
            print("End of data. Press Enter to continue.")
            has_more_data = False

        while True:
            try:
                if not has_more_data:
                    break
                # Prompt loop to capture key presses
                print("Press TAB to fetch next chunk of data, or F to save current chunk to a file")
                session.prompt()
            except KeyboardInterrupt:
                break


    
    def _query_chunks(self, arg, language):
        try:
            table = self.influxdb_client.query(query=arg, language=language, mode="chunk")
            return table
        except Exception as e:
            print(e)
            return None

    def do_write(self, arg):
        if self.active_config == {}:
            print("can't write, no active configs")
            return
        if arg == "":
            print("can't write, no line protocol supplied")
            return
        
        self.influxdb_client.write(record=arg)
    
    def do_write_file(self, args):
        if self.active_config == {}:
            print("can't write, no active configs")
            return

        temp = {}
        attributes = ['file', 'measurement', 'time', 'tags']
        temp['tags'] = []

        for attribute in attributes:
            arg_value = getattr(args, attribute)
            if arg_value is not None:
                temp[attribute] = arg_value
        if isinstance(temp['tags'], str):
            temp['tags'] =  temp['tags'].split(',')
        

        if 'measurement' in temp:
            try:
                self.influxdb_client.write_file(file=temp['file'], 
                                       measurement_name=temp['measurement'], 
                                       timestamp_column=temp['time'], 
                                       tag_columns=temp['tags'])
                
            except Exception as e:
                print(e)
        else:
            print("measurement not specified. Attempting to find measurement in file...")
            try:
                self.influxdb_client.write_file(file=temp['file'], 
                                       timestamp_column=temp['time'], 
                                       tag_columns=temp['tags'])
                
            except Exception as e:
                print(e)
                

    def do_exit(self, arg):
        'Exit the shell: exit'
        print('\nExiting ...')
        return True

    def do_EOF(self, arg):
        'Exit the shell with Ctrl-D'
        return self.do_exit(arg)

    def do_chatgpt(self, arg):
        if arg == "":
            print("can't write, no line protocol supplied")
            return
        openai_helper = OpenAIHelper()
        query = openai_helper.nl_to_sql(arg)
        print(f"Run InfluxQL query: {query}")
        self.do_query(query, language='influxql')
        

 
    def precmd(self, line):
        if line.strip() == 'sql':
            self._run_prompt_loop('(sql >) ', lambda arg: self.do_query(arg, language='sql'), 'SQL mode')
            return ''
        if line.strip() == 'influxql':
            self._run_prompt_loop('(influxql >) ', lambda arg: self.do_query(arg, language='influxql'), 'INFLUXQL mode')
            return ''
        if line.strip() == 'write':
            self._run_prompt_loop('(write >) ', self.do_write, 'write mode')
            return ''
        if line.strip() == 'chatgpt':
            self._run_prompt_loop('(chatgpt >) ', self.do_chatgpt, 'chatgpt mode')
            return ''
        return line

    def _run_prompt_loop(self, prompt, action, mode_name):
        prompt_session = self._sql_prompt_session if mode_name == 'SQL mode' else self._write_prompt_session
        while True:
            try:
                statement = prompt_session.prompt(prompt)
                if statement.strip().lower() == 'exit':
                    break
                action(statement)
            except KeyboardInterrupt:
                print(f'Ctrl-D pressed, exiting {mode_name}...')
                break
            except EOFError:
                print(f'Ctrl-D pressed, exiting {mode_name}...')
                break
    
    def create_config(self, args):
        self.config_helper._create(args)

    
    def delete_config(self, args):
        self.config_helper._delete(args)

    
    def list_config(self, args):
        self.config_helper._list(args)
    
    def use_config(self, args):
        self.config_helper._set_active(args)


    def update_config(self, args):
        self.config_helper._update(args)


        
    def _setup_client(self):
        try:
            self._database = self.active_config['database']

            self.influxdb_client = InfluxDBClient3(
                host=self.active_config['host'],
                org=self.active_config['org'],
                token=self.active_config['token'],
                database=self.active_config['database']
            )
        except Exception as e:
            print("No active config found, please run 'config' command to create a new config")


class StoreRemainingInput(argparse.Action):
    def __call__(self, parser, database, values, option_string=None):
        setattr(database, self.dest, ' '.join(values))

def parse_args():
    parser = argparse.ArgumentParser(description= _description_string
                                     )
    subparsers = parser.add_subparsers(dest='command')

    sql_parser = subparsers.add_parser('sql', help='execute the given SQL query')
    sql_parser.add_argument('query', metavar='QUERY', nargs='*', action=StoreRemainingInput, help='the SQL query to execute')
    influxql_parser = subparsers.add_parser('influxql', help='execute the given InfluxQL query')
    influxql_parser.add_argument('query', metavar='QUERY', nargs='*', action=StoreRemainingInput, help='the INFLUXQL query to execute')

    chatgpt_parser = subparsers.add_parser('chatgpt', help='execute the given chatgpt statement')
    chatgpt_parser.add_argument('query', metavar='QUERY', nargs='*', action=StoreRemainingInput, help='the chatgpt query to execute')

    write_parser = subparsers.add_parser('write', help='write line protocol to InfluxDB')
    write_parser.add_argument('line_protocol', metavar='LINE PROTOCOL',  nargs='*', action=StoreRemainingInput, help='the data to write')

    write_file_parser = subparsers.add_parser('write_file', help='write data from file to InfluxDB')
    write_file_parser.add_argument('--file', help='the file to import', required=True)
    write_file_parser.add_argument('--measurement', help='Define the name of the measurement', required=False)
    write_file_parser.add_argument('--time', help='Define the name of the time column within the file', required=True)
    write_file_parser.add_argument('--tags', help='(optional) array of column names which are tags. Format should be: ["tag1", "tag2"]', required=False)

    config_parser = subparsers.add_parser("config", help="configure the application")
    config_subparsers = config_parser.add_subparsers(dest='config_command')

    create_parser = config_subparsers.add_parser("create", help="create a new configuration")
    create_parser.add_argument("--name", help="Configuration name", required=True)
    create_parser.add_argument("--host", help="Host string", required=True)
    create_parser.add_argument("--token", help="Token string", required=True)
    create_parser.add_argument("--database", help="Database string", required=True)
    create_parser.add_argument("--org", help="Organization string", required=True)
    create_parser.add_argument("--active", help="Set this configuration as active", required=False, action='store_true')

        # Update command
    update_parser = config_subparsers.add_parser("update", help="update an existing configuration")
    update_parser.add_argument("--name", help="Configuration name", required=True)
    update_parser.add_argument("--host", help="Host string", required=False)
    update_parser.add_argument("--token", help="Token string", required=False)
    update_parser.add_argument("--database", help="Database string", required=False)
    update_parser.add_argument("--org", help="Organization string", required=False)
    update_parser.add_argument("--active", help="Set this configuration as active", required=False, action='store_true')

    # Use command
    use_parser = config_subparsers.add_parser("use", help="use a specific configuration")
    use_parser.add_argument("--name", help="Configuration name", required=True)

    delete_parser = config_subparsers.add_parser("delete", help="delete a configuration")
    delete_parser.add_argument("--name", help="Configuration name", required=True)

    list_parser = config_subparsers.add_parser("list", help="list all configurations")

    config_parser = subparsers.add_parser("help")

    return parser.parse_args()

def main():
    args = parse_args()
    app = IOXCLI()


    if args.command == 'sql':
        app.do_query(args.query, language='sql')
    if args.command == 'influxql':
        app.do_query(args.query, language='influxql')
    if args.command == 'chatgpt':
        app.do_chatgpt(args.query)
    if args.command == 'write':
        app.do_write(args.line_protocol)
    if args.command == 'write_file':
        app.do_write_file(args)
    if args.command == 'config':
        if args.config_command == 'create':
            app.create_config(args)
        elif args.config_command == 'delete':
            app.delete_config(args)
        elif args.config_command == 'list':
            app.list_config(args)
        elif args.config_command == 'update':
            app.update_config(args)
        elif args.config_command == 'use':
            app.use_config(args)
        else:
             print(_usage_string)
    if args.command == 'help':
        print(_usage_string)
    if args.command is None:
        app.cmdloop()
    

if __name__ == '__main__':
    main()

