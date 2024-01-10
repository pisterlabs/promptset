"""
Load question-answer content into Weaviate.

Currently this does a full dump-and-reload, but in future it should
continually and incrementally build the Weaviate cluster's database. 
"""

# I created a Weaviate cluster in the following way:
#
# 1. Created an account at weaviate.io; verified my email.
# 2. Clicked "Create a cluster" in the weaviate.io UI.
# 3. Selected:
#   subscription tier: sandbox
#   weaviate version: v.1.17.3
#   enable OIDC authentication: false (this data is not private)


def ingest_data(weaviate_url: str, openai_api_key: str, docs: list[str]):
    import weaviate
    from langchain.text_splitter import CharacterTextSplitter

    #TODO: Remove
    metadatas = [{"source": "https://thundergolfer.com/about"} for _ in docs]

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    #TODO: Remove   
    documents = text_splitter.create_documents(docs, metadatas=metadatas)

    client = weaviate.Client(
        url=weaviate_url,
        additional_headers={"X-OpenAI-Api-Key": openai_api_key},
    )

    client.schema.delete_all()  # drop ALL data

    client.schema.get()
    schema = {
        "classes": [
            {
                "class": "Paragraph",
                "description": "A written paragraph",
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
                        "description": "The content of the paragraph",
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
                        "name": "source",
                    },
                ],
            },
        ]
    }

    client.schema.create(schema)

    with client.batch as batch:
        for text in documents:
            batch.add_data_object(
                {"content": text.page_content, "source": str(text.metadata["source"])},
                "Paragraph",
            )


def ingest_examples(weaviate_url: str, openai_api_key: str):
    """Ingest examples into Weaviate."""
    import weaviate
    import weaviate.exceptions

    client = weaviate.Client(
        url=weaviate_url,
        additional_headers={"X-OpenAI-Api-Key": openai_api_key},
    )

    try:
        client.schema.delete_class("Rephrase")
        client.schema.delete_class("QA")
    except weaviate.exceptions.UnexpectedStatusCodeException:
        pass  # Likely failed because classes didn't already exist.
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

    client.schema.create(schema)

    documents = [
        {
            "question": "how do i load those?",
            "chat_history": "Human: What types of memory exist?\nAssistant: \n\nThere are a few different types of memory: Buffer, Summary, and Conversational Memory.",
            "answer": "How do I load Buffer, Summary, and Conversational Memory",
        },
        {
            "question": "how do i install this package?",
            "chat_history": "",
            "answer": "How do I install langchain?",
        },
        {
            "question": "how do I set serpapi_api_key?",
            "chat_history": "Human: can you write me a code snippet for that?\nAssistant: \n\nYes, you can create an Agent with a custom LLMChain in LangChain. Here is a [link](https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_agent.html) to the documentation that provides a code snippet for creating a custom Agent.",
            "answer": "How do I set the serpapi_api_key?",
        },
        {
            "question": "What are some methods for data augmented generation?",
            "chat_history": "Human: List all methods of an Agent class please\nAssistant: \n\nTo answer your question, you can find a list of all the methods of the Agent class in the [API reference documentation](https://langchain.readthedocs.io/en/latest/modules/agents/reference.html).",
            "answer": "What are some methods for data augmented generation?",
        },
        {
            "question": "can you write me a code snippet for that?",
            "chat_history": "Human: how do I create an agent with custom LLMChain?\nAssistant: \n\nTo create an Agent with a custom LLMChain in LangChain, you can use the [Custom Agent example](https://langchain.readthedocs.io/en/latest/modules/agents/examples/custom_agent.html). This example shows how to create a custom LLMChain and use an existing Agent class to parse the output. For more information on Agents and Tools, check out the [Key Concepts](https://langchain.readthedocs.io/en/latest/modules/agents/key_concepts.html) documentation.",
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

    client.schema.create(schema)

    documents = [
        {
            "question": "how do i install langchain?",
            "answer": "```pip install langchain```",
            "summaries": ">Example:\nContent:\n---------\nYou can pip install langchain package by running 'pip install langchain'\n----------\nSource: foo.html",
            "sources": "foo.html",
        },
        {
            "question": "how do i import an openai LLM?",
            "answer": "```from langchain.llm import OpenAI```",
            "summaries": ">Example:\nContent:\n---------\nyou can import the open ai wrapper (OpenAI) from the langchain.llm module\n----------\nSource: bar.html",
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