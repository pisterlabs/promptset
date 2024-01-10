import os

import weaviate
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]},
)

client.schema.get()
schema = {
    "classes": [
        {
            "class": "Rephrase",
            "description": "Rephrase Examples",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "question",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "chat_history",
                },
            ],
        },
    ]
}

try:
    client.schema.create(schema)
except weaviate.exceptions.UnexpectedStatusCodeException as e:
    if e.status_code == 422:
        print("Schema already exists, deleting and recreating")
        client.schema.delete_class("Rephrase")
        client.schema.create(schema)

documents = [
    {
        "question": "how do i define those?",
        "chat_history": "Human: What types of materializations exist?\nAssistant: \n\nThere are 4 different types of materializations: table, view, incremental and ephemeral.",
        "answer": "How do I define table, view, incremental and ephemeral materializations?",
    },
    {
        "question": "how do i install this package?",
        "chat_history": "",
        "answer": "How do I install dbt?",
    },
    {
        "question": "how do I define a test?",
        "chat_history": "Human: can you write me a code snippet for that?\nAssistant: \n\nYes, you can define a singular test in dbt using basic sql. Here is a [link](https://docs.getdbt.com/docs/build/tests) to the documentation that provides a code snippet for creating a custom test.",
        "answer": "How do I define a test?",
    },
    {
        "question": "can you write me a code snippet for that?",
        "chat_history": "Human: how do I create a snapshot?\nAssistant: \n\nTo create a snapshot in dbt, you can use the [Snapshots example](https://docs.getdbt.com/docs/build/snapshots). This example shows how to create a snapshot to handle slowly changing dimensions in dbt. For more information dbt projects, check out the [Key Concepts](https://docs.getdbt.com/docs/build/projects) documentation.",
        "answer": "Can you provide a code snippet for creating an Agent with a custom LLMChain?",
    },
]
from langchain.prompts.example_selector.semantic_similarity import sorted_values

for d in documents:
    d["content"] = " ".join(sorted_values(d))
with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            text,
            "Rephrase",
        )

client.schema.get()
schema = {
    "classes": [
        {
            "class": "QA",
            "description": "Rephrase Examples",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text",
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": False,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "content",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "question",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "answer",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "summaries",
                },
                {
                    "dataType": ["text"],
                    "description": "The link",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True,
                            "vectorizePropertyName": False,
                        }
                    },
                    "name": "sources",
                },
            ],
        },
    ]
}

try:
    client.schema.create(schema)
except weaviate.exceptions.UnexpectedStatusCodeException as e:
    if e.status_code == 422:
        print("Schema already exists, deleting and recreating")
        client.schema.delete_class("QA")
        client.schema.create(schema)

documents = [
    {
        "question": "how do i install dbt?",
        "answer": "```pip install dbt```",
        "summaries": ">Example:\nContent:\n---------\nYou can pip install dbt package by running 'pip install dbt'\n----------\nSource: foo.html",
        "sources": "foo.html",
    },
    {
        "question": "how do i create a basic model?",
        "answer": "```from langchain.llm import OpenAI```",
        "summaries": ">Example:\nContent:\n---------\nyou can create your first model by creating a simple .sql file and definint a select query\n----------\nSource: bar.html",
        "sources": "bar.html",
    },
]
from langchain.prompts.example_selector.semantic_similarity import sorted_values

for d in documents:
    d["content"] = " ".join(sorted_values(d))
with client.batch as batch:
    for text in documents:
        batch.add_data_object(
            text,
            "QA",
        )
