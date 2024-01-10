from llms.llm_base import LLMBase
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
import openai

# Queries for Neo4j database introspection
node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect(property) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output

"""

rel_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS nodeLabels, collect(property) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

rel_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
RETURN {source: label, relationship: property, target: other} AS output
"""

def schema_text(node_props, rel_props, rels):
    return f"""
        This is the schema representation of the Neo4j database.
        Node properties are the following:
        {node_props}
        Relationship properties are the following:
        {rel_props}
        Relationship point from source to target nodes
        {rels}
        Make sure to respect relationship types and directions
        """

graph2text_system = f"""
You are an assistant that helps to generate text to form nice and human understandable answers based.
The latest prompt contains the information, and you need to generate a human readable response based on the given information.
Make it sound like the information are coming from an AI assistant, but don't add any information.
Do not add any additional information that is not explicitly provided in the latest prompt.
I repeat, do not add any information that is not explicitly given.
"""

class LLMCypherRAG(LLMBase):

    def __init__(self, model:str, key:str):
        self.model = model
        openai.api_key = key
        self.driver = None
        # self.schema = self.generate_schema()

    def get_system_message(self):
        return f"""
        Task: Generate Cypher queries to query a Neo4j graph database based on the provided schema definition.
        Instructions:
        Use only the provided relationship types and properties.
        Do not use any other relationship types or properties that are not provided.
        If you cannot generate a Cypher statement based on the provided schema, explain the reason to the user.
        Schema:
        {self.schema}

        Note: Do not include any explanations or apologies in your responses.
        """
    
    def refresh_schema(self):
        self.schema = self.generate_schema()

    def generate_schema(self):
        if self.driver is None:
            return
        node_props = self.query_database(node_properties_query)
        rel_props = self.query_database(rel_properties_query)
        rels = self.query_database(rel_query)
        return schema_text(node_props, rel_props, rels)
    
    def query_database(self, neo4j_query, params={}):
        with self.driver.session() as session:
            result = session.run(neo4j_query, params)
            output = [r.values() for r in result]
            output.insert(0, result.keys())
            return output
    
    def construct_cypher(self, question, history=None):
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": question},
        ]
        # Used for Cypher healing flows
        if history:
            messages.extend(history)

        completions = openai.ChatCompletion.create(
            model=self.model,
            temperature=0.0,
            max_tokens=1000,
            messages=messages
        )
        return completions.choices[0].message.content

    def run(self, question, history=None, retry=True):
        # Construct Cypher statement
        cypher = self.construct_cypher(question, history)
        print(cypher)
        try:
            return self.query_database(cypher)
        # Self-healing flow
        except CypherSyntaxError as e:
            # If out of retries
            if not retry:
              return "Invalid Cypher syntax"
        # Self-healing Cypher flow by
        # providing specific error to GPT-4
            print("Retrying")
            return self.run(
                question,
                [
                    {"role": "assistant", "content": cypher},
                    {
                        "role": "user",
                        "content": f"""This query returns an error: {str(e)} 
                        Give me a improved query that works without any explanations or apologies""",
                    },
                ],
                retry=False
            )

    # def generate_response(self, messages):
    #     messages = [
    #         {"role": "system", "content": graph2text_system}
    #     ] + messages
    #     print(messages)
    #     # Make a request to OpenAI
    #     completions = openai.ChatCompletion.create(
    #         model=self.model,
    #         messages=messages,
    #         temperature=0.0
    #     )
    #     response = completions.choices[0].message.content
    #     print(response)
    #     # If the model apologized, remove the first line or sentence
    #     if "apologi" in response:
    #         if "\n" in response:
    #             response = " ".join(response.split("\n")[1:])
    #         else:
    #             response = " ".join(response.split(".")[1:])
    #     return response
    
    def chat_completion(self,
                        prior_messages: list[any],
                        neo4j_uri: str,
                        neo4j_user: str,
                        neo4j_password: str):
        
        if self.driver is None:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.schema = self.generate_schema()

        question = prior_messages[-1]['content']

        result = self.run(question=question, 
                          history=None, 
                          retry=True)

        return result