from cryptography.fernet import Fernet
import argparse
import sys
import keyring
import base64
import db
import synth
import disk
import openai_model
import sql_ast
import question
from tabulate import tabulate

class OpenQueryCLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            prog="openquery",
            description="Automagically generate and run SQL using natural language queries",
            usage="""openquery <command> [<args>]

The most commonly used openquery commands are:
    init        Initialize openquery
    create      Create a new resource (db, synth, model).
    delete      Delete a resource (db, synth, model).
    use         Set active resources (db, synth, model).
    list        List resource names (db, synth, model).
    ask         Ask a question
""")
        parser.add_argument(
            "command",
            help="Subcommand to run"
        )
    
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        
        getattr(self, args.command)()

    def init(self):
        print("""
       ▄▄▄·▄▄▄ . ▐ ▄ .▄▄▄  ▄• ▄▌▄▄▄ .▄▄▄   ▄· ▄▌
▪     ▐█ ▄█▀▄.▀·•█▌▐█▐▀•▀█ █▪██▌▀▄.▀·▀▄ █·▐█▪██▌
 ▄█▀▄  ██▀·▐▀▀▪▄▐█▐▐▌█▌·.█▌█▌▐█▌▐▀▀▪▄▐▀▀▄ ▐█▌▐█▪
▐█▌.▐▌▐█▪·•▐█▄▄▌██▐█▌▐█▪▄█·▐█▄█▌▐█▄▄▌▐█•█▌ ▐█▀·.
 ▀█▄▀▪.▀    ▀▀▀ ▀▀ █▪·▀▀█.  ▀▀▀  ▀▀▀ .▀  ▀  ▀ • 
                              OpenQuery - v1.0.0
        """)

        print("Creating openquery encryption key...", end="")
        key = Fernet.generate_key()
        keyring.set_password("openquery", "encryption_key", key.decode())
        print("Success!")
    
        print()
    
    def create(self):
        parser = argparse.ArgumentParser(
            prog="openquery create",
            description="Create a new resource (db, synth, model)."
        )

        parser.add_argument(
            "resource",
            help="The resource to create (db, synth, model)"
        )

        args = parser.parse_args(sys.argv[2:3])

        if args.resource == "db":
            db.create_cli()
        elif args.resource == "synth":
            synth.create_cli()
        elif args.resource == "model":
            openai_model.create_cli()

    def delete(self):
        parser = argparse.ArgumentParser(
            prog="openquery delete",
            description="Delete a resource by name."
        )

        parser.add_argument(
            "resource",
            help="The name of the resource to delete"
        )

        args = parser.parse_args(sys.argv[2:3])

        disk.delete(args.resource)

        # TODO - if the resource is active, remove it
    
    def ask(self):
        parser = argparse.ArgumentParser(
            prog="openquery ask",
            description="Ask a question and receive a SQL query. Optionally, you can opt-in to have the query automagically run by configuring it as the default behavior in your database profile."
        )

        parser.add_argument(
            "question",
            metavar="question",
            help="A natural language question for openquery to answer"
        )

        args = parser.parse_args(sys.argv[2:])
        pii_labels = question.get_pii_labels(args.question)
        if len(pii_labels) > 0:
            print("\nWarning: This question may contain personally identifiable information. Please review the following labels (in brackets) \n")
            print(question.highlight_pii(args.question, pii_labels))
            print()
            should_continue = input("Continue (y/n)? (n) ") or "n"
            if should_continue == "n":
                return

        queries = openai_model.ask(args.question)
        name = db.get_active()
        attempt = 0
        for query in queries:
            attempt += 1
            try:
                cursor = db.run_query(name, query.message.content)
                print("\n" + tabulate(cursor, headers="keys"))
                
                print("\nHere's the query I ran (attempt #{}): \n\n{}\n".format(attempt, sql_ast.standardize(query.message.content)))
                
                correctness = input("Is this correct (y/n)? (y) ") or "y"
                print()
                if correctness == "y":
                    openai_model.save_training_data(args.question, query.message.content)
                break
            except Exception as e:
                pass
        

    def use(self):
        parser = argparse.ArgumentParser(
            prog="openquery use",
            description="Set active resources (profile, synth, model, etc)"
        )

        parser.add_argument(
            "resource",
            help="The resource to use"
        )

        args = parser.parse_args(sys.argv[2:3])
        data = disk.read_bytes(args.resource)
        header = data[0]
        if header == db.DB_HEADER:
            db.activate(args.resource)
        elif header == synth.SYNTH_HEADER:
            synth.activate(args.resource)
        else:
            print("Resource {} not found".format(args.resource))

    def list(self):
        parser = argparse.ArgumentParser(
            prog="openquery list",
            description="List resource names (db, synth, model)."
        )

        print('\n'.join(disk.list_all()))

if __name__ == '__main__':
   OpenQueryCLI() 
