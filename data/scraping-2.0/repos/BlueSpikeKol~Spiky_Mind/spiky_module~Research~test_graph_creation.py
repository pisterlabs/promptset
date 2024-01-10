import os
import json
import pathlib
from pathlib import Path
from utils import config_retrieval

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI

from neo4j import GraphDatabase

import utils.openai_api.gpt_calling as gpt

PROJECT_PROMPT_TEMPLATE_1 = """
From the Project Survey below, extract the following Entities & relationships described in the mentioned format 
0. ALWAYS FINISH THE OUTPUT. Never send partial responses
1. First, look for these Entity types in the text and generate as comma-separated format similar to entity type.
   `id` property of each entity must be alphanumeric and must be unique among the entities. You will be referring this property to define the relationship between entities. Do not create new entity types that aren't mentioned below. Document must be summarized and stored inside Project entity under `summary` property. You will have to generate as many entities as needed as per the types below:
    Entity Types:
    label:'Project',id:string,name:string;summary:string //Project that is the subject of the survey; `id` property is the full name of the project, in lowercase, with no capital letters, special characters, spaces or hyphens; Contents of original document must be summarized inside 'summary' property
    label:'Goal',id:string,name:string;summary:string //Goal of the project that is the subject of the survey; `id` property is the full name of the goal, in lowercase, with no capital letters, special characters, spaces or hyphens; Contents of original document must be summarized inside 'summary' property
    label:'Technology',id:string,name:string //Technology Entity; `id` property is the name of the technology, in camel-case. Identify as many of the technologies used as possible
    label:'Restriction',id:string,name:string;//Restriction upon the completion of the project; `id` property is the name of the restriction, in camel-case; Identify as many of the restrictions as possible
    label:'Ressource',id:string,name:string;//Ressource available to the project; `id` property is the name of the ressource, in camel-case; Identify as many of the ressources as possible
    
2. Next generate each relationships as triples of head, relationship and tail. To refer the head and tail entity, use their respective `id` property. Relationship property should be mentioned within brackets as comma-separated. They should follow these relationship types below. You will have to generate as many relationships as needed as defined below:
    Relationship types:
    goal|USES_TECH|technology 
    goal|RESTRICTED|restriction
    project|HAS_RESSOURCES|ressource
    technology|RESTRICTED|restriction
    


3. The output should look like :
{
    "entities": [{"label":"Project","id":string,"name":string,"summary":string}],
    "relationships": ["goalid|USES_TECH|technologyid"]
}

Case Sheet:
$ctext
"""

PROJECT_PROMPT_TEMPLATE_2 = """
From the Project Survey below, extract the following Entities & relationships described in the mentioned format 
0. ALWAYS FINISH THE OUTPUT. Never send partial responses
1. First, look for these Entity types in the text and generate as comma-separated format similar to entity type.
   `id` property of each entity must be alphanumeric and must be unique among the entities. You will be referring this property to define the relationship between entities. Do not create new entity types that aren't mentioned below. Document must be summarized and stored inside Project entity under `summary` property. You will have to generate as many entities as needed as per the types below:
    Entity Types:
    label:'Project',id:string,name:string;summary:string //Project that is the subject of the survey; `id` property is the full name of the project, in lowercase, with no capital letters, special characters, spaces or hyphens; Contents of original document must be summarized inside 'summary' property
    label:'Goal',id:string,name:string;summary:string //Goal of the project that is the subject of the survey; `id` property is the full name of the goal, in lowercase, with no capital letters, special characters, spaces or hyphens; Contents of original document must be summarized inside 'summary' property
    label:'Technology',id:string,name:string //Technology Entity; `id` property is the name of the technology, in camel-case. Identify as many of the technologies used as possible
    label:'Restriction',id:string,name:string, value:string;//Restriction upon the completion of the project; `id` property is the name of the restriction, in camel-case; `value` property is the quantifier of the restriction or the specific restriction; Identify as many of the restrictions as possible
    label:'Ressource',id:string,name:string;//Every ressources available to the project. It can be of any type; `id` property is the name of the ressource, in camel-case; Identify as many of the ressources as possible

2. Next generate each relationships as triples of head, relationship and tail. To refer the head and tail entity, use their respective `id` property. Relationship property should be mentioned within brackets as comma-separated. They should follow these relationship types below. You will have to generate as many relationships as needed as defined below:
    Relationship types:
    goal|USES_TECH|technology 
    goal|RESTRICTED|restriction
    goal|HAS_RESSOURCES|ressource
    project|HAS_RESSOURCES|ressource
    technology|RESTRICTED|restriction



3. The output should look like :
{
    "entities": [{"label":"Project","id":string,"name":string,"summary":string}],
    "relationships": ["goalid|USES_TECH|technologyid"]
}

Case Sheet:
$ctext
"""

PROJECT_PROMPT_TEMPLATE_3 = """
From the Project Survey below, extract the following Entities & relationships described in the mentioned format 
0. ALWAYS FINISH THE OUTPUT. Never send partial responses
1. First, look for these Entity types in the text and generate as comma-separated format similar to entity type.
   `id` property of each entity must be alphanumeric and must be unique among the entities. You will be referring this property to define the relationship between entities. Do not create new entity types that aren't mentioned below. Document must be summarized and stored inside Project entity under `summary` property. You will have to generate as many entities as needed as per the types below:
    Entity Types:
    label:'Project',id:string,name:string;summary:string //Project that is the subject of the survey; `id` property is the full name of the project, in lowercase, with no capital letters, special characters, spaces or hyphens; Contents of original document must be summarized inside 'summary' property
    label:'Goal',id:string,name:string;summary:string //Goal of the project that is the subject of the survey; `id` property is the full name of the goal, in lowercase, with no capital letters, special characters, spaces or hyphens; Contents of original document must be summarized inside 'summary' property
    label:'Technology',id:string,name:string //Technology Entity; `id` property is the name of the technology, in camel-case. Identify as many of the technologies used as possible
    label:'Restriction',id:string,name:string, value:string;//Absolute restriction upon the completion of the project. Cannot have common entity with 'Concern'; `id` property is the name of the restriction, in camel-case; `value` property is the quantifier of the restriction or the specific restriction; Identify as many of the restrictions as possible
    label:'Concern',id:string,name:string, value:string;//Every non-absolute restriction that need to be taken into account while completing the project. Cannot have common entity with 'Restriction'; `id` property is the name of the concern, in camel-case; `value` property is the quantifier of the restriction or the specific concern; Identify as many of the concerns as possible
    label:'Ressource',id:string,name:string;//Every ressources available to the project. It can be of any type; `id` property is the name of the ressource, in camel-case; Identify as many of the ressources as possible

2. Next generate each relationships as triples of head, relationship and tail. To refer the head and tail entity, use their respective `id` property. Relationship property should be mentioned within brackets as comma-separated. They should follow these relationship types below. You will have to generate as many relationships as needed as defined below:
    Relationship types:
    goal|USES_TECH|technology 
    goal|RESTRICTED|restriction
    goal|CONCERNED|concern
    goal|HAS_RESSOURCES|ressource
    project|HAS_RESSOURCES|ressource
    technology|RESTRICTED|restriction
    tehnology|CONCERNED|concern



3. The output should look like :
{
    "entities": [{"label":"Project","id":string,"name":string,"summary":string}],
    "relationships": ["goalid|USES_TECH|technologyid"]
}

Case Sheet:
$ctext
"""


QUERY_PROMPT_TEMPLATE_1 = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Client, use `toLower(client.id) contains 'neo4j'`. To search for Slack Messages, use 'toLower(SlackMessage.text) contains 'neo4j'`. To search for a project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.)
5. Never use relationships that are not mentioned in the given schema
6. When asked about projects, Match the properties using case-insensitive matching and the OR-operator, E.g, to find a logistics platform -project, use `toLower(project.summary) contains 'logistics platform' OR toLower(project.name) contains 'logistics platform'`.

schema: {schema}

Examples:
Question: Which tech's goal has the greatest number of restriction?
Answer: ```MATCH (tech:Technology)<-[:USES_TECH]-(g:Goal)-[:RESTRICTED]->(restriction:Restriction)
RETURN tech.name AS Tech, COUNT(DISTINCT restriction) AS NumberOfRestriction
ORDER BY NumberOfRestriction DESC```
Question: Which goal uses the largest number of different technologies?
Answer: ```MATCH (goal:Goal)-[:USES_TECH]->(tech:Technology)
RETURN goal.name AS GoalName, COUNT(DISTINCT tech) AS NumberOfTechnologies
ORDER BY NumberOfTechnologies DESC```

Question: {question}
$ctext
"""

config = config_retrieval.ConfigManager()

llm = ChatOpenAI(
    temperature = 0,
    openai_api_key=config.openai.api_key,
    model="gpt-4"
)


CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Final answer should be easily readable and structured.
Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

cypher_prompt = PromptTemplate(
    template = QUERY_PROMPT_TEMPLATE_1,
    input_variables = ["schema", "question"]
)
def query_graph(user_input):
    graph = Neo4jGraph(url=config.neo4j.host, username=config.neo4j.user, password=config.neo4j.password)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt
        )
    result = chain(user_input)
    return result


def load_data_main_discussion(path: pathlib.Path) -> str:
    with open(path, "r") as f:
        data = json.load(f)

    result_txt = f"Project brief description: {data['form']['project_brief']}\n"

    # Get the data from the form
    index = 0
    for i in range(len(data["form"]["questions"].keys())):
        result_txt += data["form"]["questions"]["Q" + str(i)] + "\n"

        if "discussion needed" in data["form"]["answers"]["Q" + str(i)].lower():
            result_txt += data["form"]["summary_discussions"][index] + "\n"
            index += 1
        else:
            result_txt += data["form"]["answers"]["Q" + str(i)] + "\n"

    # Get the data from the main discussion
    for i in data["conversation"]["questions"].keys():
        result_txt += data["conversation"]["questions"][i] + "\n"
        result_txt += data["conversation"]["answers"][i] + "\n"

    return result_txt


def create_graph(querry: str, data: str, model: str) -> str:
    complete_query = querry.replace("$ctext", data)

    agent = gpt.GPTAgent(model)
    messages = [{"role": "user", "content": complete_query}]
    agent.messages = messages

    all_data = agent.run_agent()

    print(all_data["usage"])

    return all_data["choices"][0]["message"]["content"]


def save_graph(path: Path, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)


def generate_cypher(json_obj):
    e_statements = []
    r_statements = []

    e_label_map = {}

    # loop through our json object
    for i, obj in enumerate(json_obj):
        print(f"Generating cypher for file {i+1} of {len(json_obj)}")

        # Process entities if they exist in this part of the json_obj
        if "entities" in obj:
            for entity in obj["entities"]:
                label = entity["label"]
                id = entity["id"]
                id = id.replace("-", "").replace("_", "")
                properties = {k: v for k, v in entity.items() if k not in ["label", "id"]}

                cypher = f'MERGE (n:{label} {{id: "{id}"}})'
                if properties:
                    props_str = ", ".join(
                        [f'n.{key} = "{val}"' for key, val in properties.items()]
                    )
                    cypher += f" ON CREATE SET {props_str}"
                e_statements.append(cypher)
                e_label_map[id] = label

        # Process relationships if they exist in this part of the json_obj
        if "relationships" in obj:
            for rs in obj["relationships"]:
                src_id, rs_type, tgt_id = rs.split("|")
                src_id = src_id.replace("-", "").replace("_", "")
                tgt_id = tgt_id.replace("-", "").replace("_", "")

                src_label = e_label_map.get(src_id, "UnknownLabel")
                tgt_label = e_label_map.get(tgt_id, "UnknownLabel")

                cypher = f'MERGE (a:{src_label} {{id: "{src_id}"}}) MERGE (b:{tgt_label} {{id: "{tgt_id}"}}) MERGE (a)-[:{rs_type}]->(b)'
                r_statements.append(cypher)

    with open("cyphers.txt", "w") as outfile:
        outfile.write("\n".join(e_statements + r_statements))

    return e_statements + r_statements



def create_db(url: str, user: str, password: str, data: list) -> None:
    gdb = GraphDatabase.driver(url, auth=(user, password))

    for i, stmt in enumerate(data):
        print(f"Executing cypher statement {i + 1} of {len(data)}")
        try:
            gdb.execute_query(stmt)
        except Exception as e:
            with open("failed_statements.txt", "a") as f:
                f.write(f"{stmt} - Exception: {e}\n")


def get_data_db(template: str, data: str, model: str, url: str, user: str, password: str) -> str: #TODO replace using the langchain lib(function query_graph above)
    """
    Need to add some processing for the comprehension of the output data
    :param password:
    :param user:
    :param url:
    :param template: Template of the prompt being used to generate the cypher.
    :param data: Question from the user.
    :param model: Name of the model used.
    :return: The data from the database after executing the prompt.
    """
    query = template.replace("$ctext", data)

    agent = gpt.GPTAgent(model)
    messages = [{"role": "user", "content": query}]
    agent.messages = messages

    all_data = agent.run_agent()

    print(all_data["usage"])

    gdb = GraphDatabase.driver(url, auth=(user, password))

    q = all_data["choices"][0]["message"]["content"].replace("```", "").replace("Answer:", "").replace("cypher", "")

    print(q)

    try:
        result = gdb.execute_query(q)
        return result

    except Exception as e:
        with open("failed_statements.txt", "a") as f:
            f.write(f"{q} - Exception: {e}\n")

def create_json_from_conversation(path_folder, path_graph, ls_filename_model: list[str]) -> str:
    graph_name = None
    for filename, model, graph_name in ls_filename_model:
        d = load_data_main_discussion(path_folder / filename)
        # graph = create_graph(PROJECT_PROMPT_TEMPLATE_3, d, model)
        # save_graph(path_graph / graph_name, graph)
    return graph_name

def pipeline(path_folder, path_graph, ls_filename_model): #ls_filename_model is a list that allows testing
    filename = create_json_from_conversation(path_folder, path_graph, ls_filename_model)

    json_path = path_graph / filename
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Splitting the json_data into two separate dictionaries
    json_list = [
        {"entities": json_data.get("entities", [])},
        {"relationships": json_data.get("relationships", [])}
    ]

    d = generate_cypher(json_list)

    create_db("neo4j+s://" + config.neo4j.host, config.neo4j.user, config.neo4j.password, d)
    # test the db, no need to do previous steps if there is already a db
    query_result = query_graph("Which tech's goal has the greatest number of restriction?")
    print(query_result)

if __name__ == '__main__':
    current_script_path = Path(__file__).resolve()
    parent_folder = current_script_path.parent
    path_folder = parent_folder.joinpath('logs')
    path_graph = parent_folder.joinpath('graphs')

    # # ls_filename_model = [("logs_test_short_conversation_gpt4.json", "gpt-3.5-turbo-16k-0613", "d2_q3_short_gpt3.json"),
    #                      ("logs_test_long_conversation_gpt4.json", "gpt-3.5-turbo-16k-0613", "d2_q3_long_gpt3.json"),
    #                      ("logs_test_short_conversation_gpt4.json", "gpt-4", "d2_q3_short_gpt4.json"),
    #                      ("logs_test_long_conversation_gpt4.json", "gpt-4", "d2_q3_long_gpt4.json"), ]

    ls_filename_model = [("logs_test_short_conversation_gpt4.json", "gpt-4", "pipeline_test.json")]

    pipeline(path_folder, path_graph, ls_filename_model)

    # for filename, model, graph_name in ls_filename_model:
    #     d = load_data_main_discussion(path_folder / filename)
    #     graph = create_graph(PROJECT_PROMPT_TEMPLATE_3, d, model)
    #     save_graph(path_graph / graph_name, graph)
    #
    # # d = create_cypher(path_graph / "d2_q2_short_gpt4.json")
    # # create_db("neo4j+s://ba0b31b5.databases.neo4j.io", "neo4j", "HeSO-YmrspoQdlyGclo4mtslUcVG8cf7IRmN3bymOOQ", d)
    # print(get_data_db(QUERY_PROMPT_TEMPLATE_1, "Which tech's goal has the greatest number of restriction?",
    #                   "gpt-4", "neo4j+s://ba0b31b5.databases.neo4j.io", "neo4j",
    #                   "HeSO-YmrspoQdlyGclo4mtslUcVG8cf7IRmN3bymOOQ"))
