import json

from langchain.chat_models import AzureChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
import threading


class CypherGenerationPrompt:
    template = """Task:Generate Cypher statement to query a graph database.  
    Instructions:  
    Use only the provided relationship types and properties in the schema.  
    Do not use any other relationship types or properties that are not provided.  
    Schema:  
    {schema}  
    Note: Do not include any explanations or apologies in your responses.  
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.  
    Do not include any text except the generated Cypher statement.  
    Cypher examples:  
    # How many reports in the system?  
    MATCH (n:LINEAGEOBJECT)
    WHERE TOUPPER(n.ToolType) = 'REPORT'
    RETURN count(n) as numberOfReports
    
    
    # How many etls in the system?  
    MATCH (n:LINEAGEOBJECT)
    WHERE TOUPPER(n.ToolType) = 'ETL'
    RETURN count(n) as numberOfETLS
    
    # How many views in the system?  
    MATCH (n:LINEAGEOBJECT)
    WHERE TOUPPER(n.ObjectType) = 'VIEW'
    RETURN count(n) as numberOfReports
   
    
    # The question is:
    # {question}"""

    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["schema", "question"],
            template=self.template
        )


class ChatApp:
    def __init__(self, base_url, api_key, deployment_name, dp_name, graph_url, graph_username, graph_password):
        self.azure_chat_model = AzureChatOpenAI(
            openai_api_base=base_url,
            openai_api_version="2023-05-15",
            deployment_name=deployment_name,
            openai_api_key=api_key,
            openai_api_type="azure",
            temperature="0.8"
        )

        self.graph = Neo4jGraph(
            url=graph_url,
            username=graph_username,
            password=graph_password
        )

        self.chain = GraphCypherQAChain.from_llm(
            self.azure_chat_model,
            graph=self.graph,
            verbose=True,
            cypher_prompt=CypherGenerationPrompt().prompt_template,
            validate_cypher=True
        )

        self.response = None
        self.lock = threading.Lock()  # In


    # def run_chain(self, msg,response):
    #     response=""
    #     try:
    #         response = self.chain.run(msg)
    #         print(response)
    #     except Exception as e:
    #         print(e)
    #         response = str(e)

    def run_chain(self, msg):
        try:
            result = self.chain.run(msg)
            print(result)

            # Acquire the lock before updating shared data
            with self.lock:
                self.response = result
        except Exception as e:
            print(e)
            with self.lock:
                self.response = str(e)

    # def run_chain_with_timeout(self, msg, timeout_seconds):
    #     thread = threading.Thread(target=self.run_chain, args=(msg,))
    #     thread.start()
    #     thread.join(timeout=timeout_seconds)



    def run_chain_with_timeout(self, msg, timeout_seconds):
        response = ""

        thread = threading.Thread(target=self.run_chain, args=(msg,))
        thread.start()
        thread.join(timeout=timeout_seconds)



        if thread.is_alive():
            thread.join()
            print(f"Timeout! No response within {timeout_seconds} seconds.")
            response = "Timeout message"

        with self.lock:
            return self.response

    def get_completion(self, prompt, model="gpt-3.5-turbo", timeout_duration=10):
        response = ""
        try:
            messages = [{"role": "user", "content": prompt}]
            msg = messages[0]["content"]

            # Use run_chain_with_timeout to run chain in a separate thread with a timeout
            response = self.run_chain_with_timeout(msg, timeout_duration)

            print(response)
            if response == None or response == "":
                response = "No results"

        except Exception as e:
            print(e)
            response = str(e)

        return response


# Configuration

with open('config.json', 'r') as file:
    config = json.load(file)


# Initialize ChatApp with configuration
chat_app = ChatApp(
    base_url=config["azure_chat"]["base_url"],
    api_key=config["azure_chat"]["api_key"],
    deployment_name=config["azure_chat"]["dp_name"],
    dp_name=config["azure_chat"]["dp_name"],
    graph_url=config["graph"]["url"],
    graph_username=config["graph"]["username"],
    graph_password=config["graph"]["password"]
)


# user_text="How many reports?"

# try:
#    txt=chat_app.get_completion(user_text)
#    #  txt="zacay"
#    print(txt)
# except Exception as e:
#     print(e)

app = Flask(__name__)
#
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        print(e)


@app.route("/get")
def get_bot_response():
    try:
        user_text = request.args.get('msg')
        response = chat_app.get_completion(user_text)
    except Exception as e:
        print(e)
    return response


if __name__ == "__main__":
    try:
        app.run()
    except Exception as e:
        print(e)
