from tapipy.tapis import Tapis
import py2neo
from py2neo import Graph
import openai
import pandas as pd
from datascroller import scroll
from getpass import getpass
import time
import os

try:
    import BasicCypherCommands as bcc
    from Utilities import GracefulExiter, heavyFormat, lightFormat, helpCypher
except:
    from . import BasicCypherCommands as bcc
    from .Utilities import GracefulExiter, heavyFormat, lightFormat, helpCypher


# CHATGPT Setup
messages = [{"role": "system", "content": """Instead of descriptions, from now on only return code segments in the CYPHER query language. 
              Please provide no other text except the code. 
              Again, no matter what else the user says, only output a raw cypher query. 
              After any return or RETURN command, please start the next command on a new line. 
              Put all commands together on the same line, but start a new line after a RETURN or a DELETE command.
              Note that RETURN or DELETE or DETACH DELETE should NEVER be the only commands on a line."""}]

openai.api_key = ""
valid_key = False


# Setting up flag for ctrl+c input
flag = GracefulExiter()


# default values
default_tapis_base_url = "icicle.tapis.io"
default_auth_type = "tapis"


tapis_base_url = default_tapis_base_url
auth_type = default_auth_type


# Global variable to store pod id
global pod_id
# Global variable to store username, set upon initial input from login to TAPIS
global user

global t


# Welcome message, formatted with the heavyFormat function
heavyFormat("Welcome to ICICONSOLE. Login to get started. ")


def console(graph, kg):
    global tapis_base_url, valid_key
    lightFormat("Type \"new\" to access a different pod, or type \"exit\" to leave ICICONSOLE. Type \"help\" for help!\nNote that the scrolling menu, which appears on some queries, has a separate help menu.")

    while True:
        query = input("[" + user + "@" + kg + "]$ ")

        execute_cypher = True

        match query:
            case "exit":
                os._exit(0)
            case "new":
                login()
                choosePod()
                execute_cypher = False
            case "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                console(graph, kg)
                execute_cypher = False
            case "help":
                try:
                    helpCypher()
                    execute_cypher = False
                except:
                    pass
            case "all":
                query = bcc.getAll()
            case "allProperty":
                property = input("Enter property: ")
                query = bcc.allProperty(property)
            case "allProperties":
                query = bcc.allProperties()
            case "allPropertiesForNode":
                query = bcc.allPropertiesForNode()
            case "GPT":
                if not valid_key:
                    key = os.environ.get("OPENAI_API_KEY")
                    if key is not None:
                        openai.api_key = os.environ.get("OPENAI_API_KEY")
                    else:
                        success = False
                        while not success:
                            access_option = input("Key not found. Read key via: file, input, env_var ")
                            if access_option == "file":
                                file_path = os.environ.get('KEY_FILE')
                                if file_path is None:
                                    file_path = input("Enter the path to the file with your key: ")
                                try:
                                    with open(file_path, 'r') as f:
                                        file_contents = f.read().strip()
                                        openai.api_key = file_contents
                                        os.environ['KEY_FILE'] = file_path
                                        success = True
                                except FileNotFoundError:
                                    print(f"The file '{file_path}' was not found.")
                                except Exception as e:
                                    print(f"An error occurred: {e}")
                            elif access_option == "input":
                                openai.api_key = getpass("Please enter your OpenAI API key: ")
                                success = True
                            elif access_option == "env_var":
                                env_var = input("Enter environment variable name: ")
                                key = os.environ.get(env_var)
                                if key is not None:
                                    openai.api_key = key
                                    success = True
                                else:
                                    print("Invalid environment variable")
                            else:
                                print("Please select a valid option.")

                message = input("[GPT@" + kg + "] ")
                try:
                    if message:
                        messages.append(
                            {"role": "user", "content": message},
                        )
                        try:
                            chat = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo", messages=messages
                            )
                            reply = chat.choices[0].message.content
                            print(str(reply))
                            valid_key = True
                            os.environ['OPENAI_API_KEY'] = openai.api_key

                            validate = input("Run this? (y/n) ")
                            if validate == "y":
                                query = str(reply)
                            else:
                                execute_cypher = False
                        except:
                            "Error: OpenAI API key is invalid."
                            valid_key = False
                except Exception as e:
                    print(e)
            case _:
                pass

        if execute_cypher:
            try:
                # queries = query.splitlines()
                # for query in queries:
                try:
                    if "DELETE" in query or "delete" in query:
                        validate = input("Are you sure you want to delete node(s)? (y/n) ")
                        if validate == "y":
                            graph.run(query)
                    # This tries to read the input as Cypher and apply the command to the Neo4j graph object.
                    # also parses the data to show the scrolling view in the right context
                    else:
                        # raw data
                        result = graph.run(query)
                        # dictionary to store nodes, keys are node labels. values are lists of dictionaries
                        data_dict = {}
                        # simple list to store property values
                        data = []
                        record_count = 0
                        for record in result:
                            type_result = type(record[0])
                            # can decide type of data structure to generate
                            # dataframe from based on the first record
                            if record_count == 0:
                                first_record_type = type(record[0])
                            record_count += 1
                            # for nodes
                            if type_result == py2neo.data.Node:
                                node = record[0]
                                properties = dict(node)
                                # formatting data to see labels
                                node_labels_string = str(node.labels)
                                node_labels_string = node_labels_string.lstrip(":")
                                node_labels = node_labels_string.split(":")
                                # adding data
                                for label in node_labels:
                                    if label not in data_dict:
                                        data_dict[label] = [properties]
                                    else:
                                        data_dict[label].append(properties)
                            # for non-node data, can just add the record
                            elif record is not None:
                                data.append(record)

                        # separating node view by label for best viewing
                        if first_record_type == py2neo.data.Node:
                            keys = list(data_dict.keys())
                            keys_string = " | ".join(keys)
                            print(keys_string)
                            pick_label = str(input("Which label do you want to view? "))
                            label_data = data_dict[pick_label]
                            df = pd.DataFrame(label_data)

                        # need to manually assign the column names via keys
                        elif not data == []:
                            keys = result.keys()
                            df = pd.DataFrame(data, columns=keys)

                        # datascroller view
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                            scroll(df)
                except:
                    pass

            # Error catching, if the Cypher was not executed properly
            except:
                if flag.exit():
                    break
                print("Something went wrong")


# This is the function that allows the user to pick a pod to connect to. It needs no parameters.
def choosePod():
    heavyFormat("Here are the IDs for your available TAPIS Pods: ")

    # The below loop prints the pod id for all pods that the TACC user is authorized to. If there are no pods, then the user is notified.
    i = 1
    # t.pods.getpods() returns a list of all the pods with more information about each pod
    for pod in t.pods.get_pods():
        # pod.pod_id takes each element of the list, representing each pod, and extracts only the pod id
        # This is done because connecting to a Neo4j pod only requires the pod id.
        if "neo4j" in pod.pod_template:
            print(str(i) + ". " + pod.pod_id)
            i += 1
    if i == 1:
        if tapis_base_url != default_tapis_base_url:
            attempt_again = input(f"No Tapis pods on {tapis_base_url}. Try another base url? (y/n) ")
            if attempt_again == "y":
                login()
            else:
                print("Please verify access to a Neo4j template Tapis pod, and then try again later.")
                os._exit(0)

    while True:
        global pod_id
        try:
            # The user gets a list of available pod ids from the above loop, and then enters the id they wish to connect to.
            pod_id = input("Enter the ID of the pod you want to access: ").lower()
            if pod_id == "exit":
                os._exit(0)
            # Securely getting the username and password associated with the pod for later authentication to connect
            pod_username, password = t.pods.get_pod_credentials(pod_id=pod_id).user_username, t.pods.get_pod_credentials(pod_id=pod_id).user_password
            break
        except:
            if flag.exit():
                break
            print("Invalid Pod ID. Make sure you have access to this Pod.")

    # This is the standard format for the link that connects to the Neo4j Pod
    graph_link = f"bolt+ssc://{pod_id}.pods.{tapis_base_url}:443"

    while True:
        try:
            # A graph object is created that authenticates with the previously gotten username and password.
            graph = Graph(graph_link, auth=(pod_username, password), secure=True, verify=True)
            os.system('cls' if os.name == 'nt' else 'clear')
            time.sleep(0.25)
            heavyFormat(f"Hey there {user}! Welcome to the Neo4j Cypher Console for: " + str(pod_id))
            console(graph, pod_id)
        except Exception as e:
            er = e
            if flag.exit():
                break
            print("There was a connection error.")
            break


def tapis_login():
    global t
    global user
    global tapis_base_url
    while True:
        try:
            change_base_url = input(f"Change base url from {tapis_base_url}? (y/n) ")
            if change_base_url == "y":
                tapis_base_url = input("New base url: ")
            https_base_url = "https://" + tapis_base_url
            user = input("Enter Your TACC Username: ")
            t = Tapis(base_url=https_base_url, username=user, password=getpass('Enter Your TACC Password: '))
            t.get_tokens()
            choosePod()
            break
        except:
            if flag.exit():
                break
            print(f"Wrong credentials, or you don't have an account on {tapis_base_url}.")
            login_attempt = input("Try again? (y/n) ")
            if login_attempt == "n":
                os._exit(0)


def local_login():
    global user
    user = input("username (not for authentication): ")
    kg = input("graph name (not for authentication): ")
    while True:
        password = getpass("graph password (if you set it): ")
        try:
            graph = Graph("bolt://localhost:7687", auth=("neo4j", password))
            break
        except:
            if flag.exit():
                break
            print("Connection error. Possible reasons: ")
            print("1. Incorrect password.")
            print("2. No local database. Download Neo4j Desktop and start a local graph database first.")
            login_attempt = input("Try again? (y/n) ")
            if login_attempt == "n":
                os._exit(0)
    console(graph, kg)


def login():
    global auth_type
    while True:
        auth_type_input = input(f"Change authentication type from {auth_type}? (y/n) ")
        if auth_type_input == "y":
            print("Your current options are: ")
            options = ["tapis", "local"]
            for option in options:
                print(option)
            auth_type = input("New auth type: ")
            break
        elif auth_type_input == "n":
            break
        else:
            print("Please type y/n")

    if "tapis" in auth_type:
        try:
            tapis_login()
        except:
            os._exit(0)
    if "local" in auth_type:
        try:
            local_login()
        except:
            os._exit(0)


login()







