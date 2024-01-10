import logging
import os
from app import neo4jdb
from neo4j.exceptions import CypherSyntaxError
from openai import OpenAI
import json


def prettyData(data):
    return json.dumps([row.tolist() for row in data])


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


class Schema:

    def __init__(self, schema):
        host = neo4jdb.setSchema(schema)
        self.neo4j = neo4jdb.Neo4jDB(host)
        self.schema = self.generateSchema()
        logging.info(f'The schema generated is:{self.schema}')
        self.client = OpenAI()
        self.client.api_key = os.getenv("OPENAI_API_KEY")

    def nodeProperties(self):
        return self.neo4j.run_query_no_df("""
        CALL apoc.meta.data()
        YIELD label, other, elementType, type, property
        WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
        WITH label AS nodeLabels, collect(property) AS properties
        RETURN {labels: nodeLabels, properties: properties} AS output
        """)

    def relProperties(self):
        return self.neo4j.run_query_no_df("""
        CALL apoc.meta.data()
        YIELD label, other, elementType, type, property
        WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
        WITH label AS nodeLabels, collect(property) AS properties
        RETURN {type: nodeLabels, properties: properties} AS output
        """)

    def rel(self):
        return self.neo4j.run_query_no_df("""
        CALL apoc.meta.data()
        YIELD label, other, elementType, type, property
        WHERE type = "RELATIONSHIP" AND elementType = "node"
        RETURN {source: label, relationship: property, target: other} AS output
        """)

    def generateSchema(self):
        node_props = self.nodeProperties()
        rel_props = self.relProperties()
        rels = self.rel()

        return schema_text(node_props, rel_props, rels)

    def refreshSchema(self):
        self.schema = self.generateSchema()

    def getSystemMessage(self):
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

    def constructCypher(self, question, history=None):
        messages = [
            {"role": "system", "content": self.getSystemMessage()},
            {"role": "user", "content": question},
        ]
        # Used for Cypher healing flows
        if history:
            messages.extend(history)

        completions = self.client.chat.completions.create(
            model="gpt-4",
            temperature=0.0,
            max_tokens=1000,
            messages=messages
        )
        return completions.choices[0].message.content

    def queryDatabase(self, query, params={}):
        result = self.neo4j.run_query(query, params)
        keys = [r for r in result]
        values = [r for r in result.values]

        return [keys, values]

    # The run function starts by generating a Cypher statement. Then, the generated Cypher statement is used to query
    # the Neo4j database. If the Cypher syntax is valid, the query results are returned. However, suppose there is a
    # Cypher syntax error. In that case, we do a single follow-up to GPT-4, provide the generated Cypher statement it
    # constructed in the previous call, and include the error from the Neo4j database. GPT-4 is quite good at fixing a
    # Cypher statement when provided with the error.
    # note: The self-healing Cypher flow has only one iteration, ff the follow-up doesn't provide a valid Cypher
    # statement, the function returns the "Invalid Cypher syntax response"
    def run(self, question, history=None, retry=True):
        # Construct Cypher statement
        cypher = self.constructCypher(question, history)
        logging.info('Cypher generated:')
        logging.info(cypher)
        try:
            return self.queryDatabase(cypher)
        # Self-healing flow
        except CypherSyntaxError as e:
            # If out of retries
            if not retry:
                return "ERROR: Invalid Cypher syntax"
            # Self-healing Cypher flow by
            # providing specific error to GPT-4
            logging.info("Retrying...")
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

    def ask(self, question, retry=True):
        answer = self.run(question, retry=retry)
        if type(answer[1]) is str:
            return answer                   # an error occurred no answer
        else:
            logging.info(f'The answer is: \n {answer[0]}\n {answer[1]}')

            return [answer[0], prettyData(answer[1])]
