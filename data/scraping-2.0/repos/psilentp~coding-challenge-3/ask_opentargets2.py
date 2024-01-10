import json
import requests
from utils import extract_values
import openai
import os

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

ROLE_ = "role"
CONTENT_ = "content"
ASSISTANT_ = "assistant"
SYSTEM_ = "system"
USER_ = "user"

openai.api_key = os.environ.get("OPENAI_API_KEY")


def list_schema_types():
    base_url = GRAPHQL_URL
    query = """{
      __schema {
        types {
          name,
          description
        }
      }
    }"""
    try:
        response = requests.post(base_url, json={"query": query})
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return json.loads(response.text)


system_message = [
    {
        ROLE_: SYSTEM_,
        CONTENT_: f"""
      You are a code assistant that will generate GraphQL queries.
      Follow previous responses closely. Only output graphql queries. 
      Do not add any extra text to the response.
      Use the types from:
      {list_schema_types()}
      """,
    }
]

example_1 = [
    {ROLE_: USER_, CONTENT_: "What are the targets of vorinostat"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_drug_and_extract_target {
  search(queryString: "vorinostat", entityNames: "drug") {
    hits {
      object {
        ...extract_target_info
      }
    }
  }
}

fragment extract_target_info on Drug {
  linkedTargets {
    rows {
      _extracted:approvedSymbol
    }
  }
}""",
    },
]

example_2a = [
    {ROLE_: USER_, CONTENT_: "What are the top 3 diseases associated with ABCA4?"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_target_and_extract_disease {
  search(queryString: "ABCA4", entityNames: "target") {
    hits {
      object {
        ...extract_disease_info
      }
    }
  }
}

fragment extract_disease_info on Target {
  associatedDiseases(page: {index: 0, size: 3}) {
    rows {
      disease { _extracted:name }
    }
  }
}
""",
    },
]

example_2b = [
    {ROLE_: USER_, CONTENT_: "What are the diseases associated with ABCA4?"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_target_and_extract_disease {
  search(queryString: "ABCA4", entityNames: "target") {
    hits {
      object {
        ...extract_disease_info
      }
    }
  }
}

fragment extract_disease_info on Target {
  associatedDiseases {
    rows {
      disease { _extracted:name }
    }
  }
}
""",
    },
]

example_3 = [
    {
        ROLE_: USER_,
        CONTENT_: "Find drugs that are used for treating ulcerative colitis.",
    },
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_disease_and_extract_known_drugs {
  search(queryString: "ulcerative colitis", entityNames: "disease") {
    hits {
      object {
        ...extract_drug_info
      }
    }
  }
}

fragment extract_drug_info on Disease {
  knownDrugs {
    rows {
      drug { _extracted:name }
    }
  }
}
""",
    },
]

example_4 = [
    {ROLE_: USER_, CONTENT_: "what are the genes associated with Alzheimer's"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_disease_and_extract_genes {
  search(queryString: "Alzheimer's", entityNames: "disease") {
    hits {
      object {
        ...extract_gene_info
      }
    }
  }
}

fragment extract_gene_info on Disease {
  associatedTargets {
    rows {
      target { _extracted:approvedSymbol }
    }
  }
}
""",
    },
]

example_4b = [
    {ROLE_: USER_, CONTENT_: "what are the pathways associated with Alzheimer's"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_disease_and_extract_pathway {
  search(queryString: "Alzheimer's", entityNames: "disease") {
    hits {
      object {
        ...extract_pathway_info
      }
    }
  }
}

fragment extract_pathway_info on Disease {
  associatedTargets {
    rows {
      target { _extracted:approvedSymbol }
    }
  }
}
""",
    },
]

example_5 = [
    {ROLE_: USER_, CONTENT_: "what are the side effects of thalidomide?"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_drug_and_extract_side_effects{
  search(queryString: "thalidomide", entityNames: "drug") {
    hits {
      object {
        ...extract_adverse_events_info
      }
    }
  }
}

fragment extract_adverse_events_info on Drug {
  drugWarnings{
    _extracted:toxicityClass
  }
  adverseEvents {
    rows {
      _extracted:name
    }
  }
}
""",
    },
]

example_6 = [
    {ROLE_: USER_, CONTENT_: "Where is GFAP expressed?"},
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
# single level query
query search_for_gene_and_extract_expression {
  search(queryString: "GFAP", entityNames: "targert") {
    hits {
      object {
        ...extract_expression_info
      }
    }
  }
}

fragment extract_expression_info on Target {
  expressions {
    tissue {_extracted:label}
    }
  }
""",
    },
]

example_7 = [
    {
        ROLE_: USER_,
        CONTENT_: "Which diseases are associated with the genes targetted by fasudil?",
    },
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
query search_for_drug_and_extract_target {
  search(queryString: "fasudil", entityNames: "drug") {
    hits {
      object {
        ...extract_linked_targets
      }
    }
  }
}

fragment extract_linked_targets on Drug {
  linkedTargets {
    rows {
      ...extract_disease_info
    }
  }
}

fragment extract_disease_info on Target {
  associatedDiseases {
    rows {
      disease { _extracted:name }
    }
  }
}
""",
    },
]

example_8 = [
    {
        ROLE_: USER_,
        CONTENT_: "Show all the diseases that have at least 5 pathways associated with Alzheimer",
    },
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
query search_for_disease_and_extract_targets {
  search(queryString: "Alzheimer", entityNames: "disease") {
    hits {
      object {
        ...extract_associated_targets
      }
    }
  }
}

fragment extract_associated_targets on Disease {
  associatedTargets(page: { index: 0, size: 5 }) {
    rows {
      target {
        ...extract_disease_info
      }
    }
  }
}

fragment extract_disease_info on Target {
  associatedDiseases {
    rows {
      disease {
        _extracted: name
      }
    }
  }
}
""",
    },
]

example_9 = [
    {
        ROLE_: USER_,
        CONTENT_: "What are the drugs that interact with the top 3 genes associated with Cystic Fibrosis",
    },
    {
        ROLE_: ASSISTANT_,
        CONTENT_: """
query search_for_disease_and_extract_target {
  search(queryString: "Cystic Fibrosis", entityNames: "disease") {
    hits {
      object {
        ...extract_associated_genes
      }
    }
  }
}

fragment extract_associated_genes on Disease {
  associatedTargets(page: {index: 0, size: 3}) {
    rows {
      target {
        ...extract_known_drugs
      }
    }
  }
}

fragment extract_known_drugs on Target {
  knownDrugs {
    rows {
      drug {
        _extracted: name
      }
    }
  }
}
""",
    },
]


def list_fields_for_schema_types(opentargets_graphql_type):
    """introspect the graphql schema to determine reference fields"""
    base_url = "https://api.platform.opentargets.org/api/v4/graphql"
    query = (
        """{
    __type(name: "%s") {
    fields {
      name,
      description,
    }
  }
}"""
        % opentargets_graphql_type
    )

    try:
        response = requests.post(base_url, json={"query": query})
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    return response.text


# user_input = "what are the side effects of prednisone?"
# user_input ="Find the diseases associated with BRCA1"
# user_input = "Find drugs that are used for treating ulcerative colitis"
# user_input = "where is BCRA2 expressed?"
# user_input = "Which diseases are associated with the genes targetted by fasudil?"
# user_input = "Show all the diseases that have at least 5 pathways associated with Alzheimer's."
# user_input = "What are the drugs that interact with the top 2 targets associated with Heart Disease"
# user_input = input("How can I help you today?\n")

query = [{ROLE_: USER_, CONTENT_: user_input}]
messages = (
    system_message
    + example_1
    + example_2a
    + example_2b
    + example_3
    + example_4
    + example_4b
    + example_5
    + example_6
    + example_7
    + example_8
    + example_9
)


def completion_with_schema_enrichment(messages, query):
    functions = [
        {
            "name": "list_fields_for_schema_types",
            "description": "introspect the graphql schema to determine reference fields",
            "parameters": {
                "type": "object",
                "properties": {
                    "opentargets_graphql_type": {
                        "type": "string",
                        "description": "the GraphQL type to inspect for linked fields",
                    },
                },
                "required": ["opentargets_graphql_type"],
            },
        }
    ]
    messages = messages.copy()
    for populate in range(1):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages + query,
            temperature=0,
            max_tokens=250,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            functions=functions,
            function_call={"name": "list_fields_for_schema_types"},
            stop=["###"],
        )
        response_message = response["choices"][0]["message"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = list_fields_for_schema_types(**function_args)
        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": "list_fields_for_schema_types",
                "content": function_response,
            }
        )

    final_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages + query,
        temperature=0,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=-2,
        stop=["###"],
    )
    return final_response


def bare_completion(messages, query):
    final_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages + query,
        temperature=0,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["###"],
    )
    return final_response


try:
    final_response = bare_completion(messages, query)
    response_text = final_response["choices"][0]["message"]["content"]
    # print(response_text)
    base_url = GRAPHQL_URL
    response = requests.post(base_url, json={"query": response_text})
    response.raise_for_status()
except Exception:
    final_response = completion_with_schema_enrichment(messages, query)
    response_text = final_response["choices"][0]["message"]["content"]
    # print(response_text)
    base_url = GRAPHQL_URL
    response = requests.post(base_url, json={"query": response_text})
    response.raise_for_status()

api_response = json.loads(response.text)
hits_list = api_response["data"]["search"]["hits"][0]


extracted_values = extract_values(hits_list, search_key="_extracted")

for i, j in enumerate(extracted_values):
    print(f"{i+1}. {j}")
