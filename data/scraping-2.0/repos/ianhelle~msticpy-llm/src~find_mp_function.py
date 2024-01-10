# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
This module provides a class for searching for MSTICPy functions
from a natural language question.

Classes:
    MPFuncSearch: A class for searching for MSTICPy functions
    from a natural language question.

Functions:
    create_qa_prompt: A function that creates a prompt for a
        question-answering model.
    enumerate_mp_functions: A function that enumerates MSTICPy functions.
    func_df_to_vector_store: A function that converts a DataFrame of MSTICPy
        functions to a vector store.

"""
import json
from typing import Dict, List, Optional, Union

from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import pandas as pd
from IPython.display import Markdown
import msticpy as mp
from msticpy.datamodel import entities
from msticpy.init.pivot_core.pivot_container import PivotContainer

from _version import VERSION

__version__ = VERSION
__author__ = "Ian Hellen"


_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, don't try to make up an answer, just return
null as the matched function.

1. Find the most relevant function that could answer the Question
   - favor functions of type "pivot" and "pandas_ext" over functions of type "query"
     if there are relevant matches
   - only in the case Threat Intelligence/IoC functions, favor functions
     of the form <entity>.tilookup_* if they exist in the context below.
2. Suggest up to {num_functions} additional functions that might be relevant to the question

Output the answer as a JSON dictionary with the following fields:
- "function": the name of the function (or null if no function was found)
- "additional_functions": a list of up to {num_functions} additional functions that might
   be relevant to the question (or [] if no additional functions were found)

{{context}}

Question: {{question}}

"""


class MPFuncSearch:
    """Search for MSTICPy functions for a natural language question."""

    NUM_RETRIEVALS = 15

    def __init__(
        self,
        qry_providers: Optional[Union[mp.QueryProvider, List[mp.QueryProvider]]] = None,
    ):
        """
        Initialize the retriever.

        Parameters
        ----------
        qry_providers : Union[mp.QueryProvider, List[mp.QueryProvider]], optional
            The query provider(s) to use for function enumeration, by default None.

        """
        self.rebuild_index(qry_providers)

    def rebuild_index(
        self,
        qry_providers: Optional[Union[mp.QueryProvider, List[mp.QueryProvider]]] = None,
    ):
        """
        Rebuild the index for the retriever.

        Parameters
        ----------
        qry_providers : Union[mp.QueryProvider, List[mp.QueryProvider]], optional
            The query provider(s) to use for function enumeration, by default None.

        """
        if isinstance(qry_providers, mp.QueryProvider):
            self.qry_providers = [qry_providers]
        else:
            self.qry_providers = qry_providers or [mp.QueryProvider("MSSentinel_New")]
        self.comb_funcs_df = enumerate_mp_functions(self.qry_providers)
        self.vector_store = func_df_to_vector_store(self.comb_funcs_df, "search_text")

    def find_mp_func_names(self, question, num_functions: int = 5) -> str:
        """
        Return the most relevant function for the question.

        Parameters
        ----------
        question : str
            The question to answer.
        num_functions : int, optional
            The number of functions to return, by default 5

        Returns
        -------
        str
            A string containing the formatted response.

        """
        mp_func_prompt = create_qa_prompt(
            self.vector_store, num_functions=num_functions
        )
        response = mp_func_prompt(question)
        return self._unpack_llm_response(response)

    def find_mp_function(
        self,
        question,
        num_functions: int = 5,
        docstring: bool = False,
        markdown: bool = True,
    ) -> Union[str, Markdown]:
        """
        Find the most relevant function that could answer the question.

        Parameters
        ----------
        question : str
            The question to answer.
        num_functions : int, optional
            The number of functions to return, by default 5
        docstring : bool
            A boolean indicating whether to include the function's
            docstring in the response, by default False
        markdown : bool
            A boolean indicating whether to return the response as a Markdown
            object, by default False

        Returns
        -------
        Union[str, Markdown]
            A string or Markdown object containing the formatted response.

        """
        mp_func_prompt = create_qa_prompt(
            self.vector_store, num_functions=num_functions
        )
        response = mp_func_prompt(question)
        func_dict = self.find_mp_func_names(response)
        if not func_dict:
            return "No function found - try rephrasing the question."
        return self.display_mp_func_details(
            func_dict["function"],
            func_dict["additional_functions"],
            docstring=docstring,
            markdown=markdown,
        )

    def find_mp_function_full(
        self,
        question,
        num_functions: int = 5,
        docstring: bool = True,
        markdown: bool = True,
    ) -> Markdown:
        """
        Find the most relevant function that could answer the question.

        Parameters
        ----------
        question : str
            The question to answer.
        num_functions : int, optional
            The number of functions to return, by default 5
        docstring : bool
            A boolean indicating whether to include the function's
            docstring in the response, by default True
        markdown : bool
            A boolean indicating whether to return the response as a Markdown
            object, by default True

        Returns
        -------
        Union[str, Markdown]
            A string or Markdown object containing the formatted response.

        """
        return self.find_mp_function(
            question=question,
            num_functions=num_functions,
            docstring=docstring,
            markdown=markdown,
        )

    def display_mp_func_details(
        self,
        function_name: str,
        additional_functions: Optional[List[str]] = None,
        docstring=False,
        markdown=False,
    ):
        """
        Return the details of the function as Markdown.

        Parameters
        ----------
        function_name : str
            The name of the function to display details for.
        additional_functions : Optional[List[str]], optional
            A list of additional functions that might be relevant
            to the question, by default None
        docstring : bool, optional
            A boolean indicating whether to include the function's docstring
            in the response, by default False
        markdown : bool, optional
            A boolean indicating whether to return the response as a Markdown
            object, by default True

        Returns
        -------
        str
            The details of the function as Markdown.
        """
        data_match = self.comb_funcs_df[
            self.comb_funcs_df["function_name"] == function_name
        ]
        if data_match.empty:
            print("Could not find suitable function.")
            return None
        return format_response(
            data_match.iloc[0],
            additional_functions,
            docstring=docstring,
            markdown=markdown,
        )

    def _unpack_llm_response(self, response) -> Optional[Dict[str, Union[str, List[str]]]]:
        answer = response["result"]
        if answer == "I don't know":
            print("Could not find suitable function.")
            return None
        if answer.strip().startswith("Answer:"):
            answer = answer.replace("Answer:", "").strip()
        return json.loads(answer)


# ---------------------------------------
# Functions to retrieve MSTICPy functions
# ---------------------------------------


def get_entity_classes():
    """Return a dictionary of entity names and classes."""
    entity_classes = {}
    for item_name in dir(entities):
        item = getattr(entities, item_name)
        if isinstance(item, type) and issubclass(item, entities.Entity):
            entity_classes[item.__name__] = item
    return entity_classes


def get_pivot_func(entity, func_name):
    """Get the pivot function for an entity and pivot name."""
    func_parts = func_name.split(".")
    curr_node = entity
    for func_part in func_parts:
        curr_node = getattr(curr_node, func_part)
        if isinstance(curr_node, PivotContainer):
            continue
        elif callable(curr_node):
            return {
                "func": curr_node,
                "name": func_name,
                "description": curr_node.__doc__.strip().split("\n")[0],
                "docstring": curr_node.__doc__.strip(),
            }
    print(func_name, func_parts, curr_node, entity)
    raise TypeError("no pivot function found")


def get_pivot_functions():
    """Return a dataframe of pivot functions."""
    entity_classes = get_entity_classes()
    entity_pivot_funcs = {}
    for entity in entity_classes.values():
        pivot_dict = {pivot: get_pivot_func(entity, pivot) for pivot in entity.pivots()}
        if pivot_dict:
            entity_pivot_funcs[entity.__name__] = pivot_dict

    pivot_list = []
    for entity in entity_classes.values():
        for pivot_name in entity.pivots():
            pivot_props = get_pivot_func(entity, pivot_name)
            pivot_list.append(
                {
                    "type": "pivot",
                    "function_name": f"{entity.__name__}.{pivot_name}",
                    "example": f'{entity.__name__}.{pivot_name}("<value>"|<list of values>)',
                    "entity": entity.__name__,
                    "description": pivot_props["description"],
                    "docstring": pivot_props["docstring"],
                }
            )

    return pd.DataFrame(pivot_list)


def get_pandas_ext_methods():
    """Return a dataframe of pandas extension methods."""
    pd_ext_methods = []
    data_frame = pd.DataFrame()
    for attrib in (attrib for attrib in dir(data_frame) if attrib.startswith("mp")):
        ext_attrib = getattr(data_frame, attrib)
        for func_name in (
            attrib for attrib in dir(ext_attrib) if not attrib.startswith("_")
        ):
            doc = getattr(ext_attrib, func_name).__doc__
            pd_ext_methods.append(
                {
                    "type": "pandas_ext",
                    "function_name": f"df.{attrib}.{func_name}",
                    "example": f"df.{attrib}.{func_name}(<args...)",
                    "description": doc.strip().split("\n")[0],
                    "docstring": doc,
                }
            )

    return pd.DataFrame(pd_ext_methods)


_GENERIC_QUERY_PARAMS = [
    "start",
    "end",
    "add_query_items",
    "time_column",
    "query_project",
    "table",
]


def get_query_functions(qry_prov):
    """Return a dataframe of query functions."""
    query_desc = [
        {
            "type": "query",
            "function_name": f"{query}",
            "example": f"qry_prov.{query}(start=<start>, end=<end>, <parameters...>)",
            "description": qry_prov.query_store[query].description,
            "docstring": qry_prov.query_store[query].create_doc_string(),
            "parameters": ", ".join(
                [
                    param
                    for param in qry_prov.query_store[query].params
                    if param not in _GENERIC_QUERY_PARAMS
                ]
            ),
            "additional_parameters": ", ".join(
                [
                    param
                    for param in qry_prov.query_store[query].params
                    if param in _GENERIC_QUERY_PARAMS
                ]
            ),
        }
        for query in qry_prov.list_queries()
    ]
    return pd.DataFrame(query_desc)


# ## Combine dataframes


def enumerate_mp_functions(qry_providers: Optional[List[mp.QueryProvider]]):
    """Return a dataframe of MSTICPy functions."""
    pivot_df = get_pivot_functions()
    pd_ext_methods_df = get_pandas_ext_methods()

    query_func_list = [get_query_functions(qry_prov) for qry_prov in qry_providers]
    query_desc_df = pd.concat(query_func_list, ignore_index=True)
    return combine_func_dfs([pd_ext_methods_df, query_desc_df, pivot_df])


def combine_func_dfs(df_list):
    """Return combined func dataframe with 'search_text' column."""
    comb_df = pd.concat(df_list, ignore_index=True)
    comb_df.head()
    comb_df["search_text"] = (
        "function_name: "
        + comb_df["function_name"]
        + ", type: "
        + comb_df["type"]
        + ", description: "
        + comb_df["description"]
        # + ", docstring: "
        # + comb_df["docstring"]
    )
    return comb_df


# ----------------------------------------------
# Create OpenAI embeddings
# ----------------------------------------------


def func_df_to_vector_store(data, column):
    """Get documents from web pages."""
    loader = DataFrameLoader(data, page_content_column=column)
    raw_documents = loader.load()
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(raw_documents, embeddings)


def create_qa_prompt(vector_store, num_functions=5, num_retrievals=15):
    """Create a prompt for the OpenAPI model model."""
    prompt = PromptTemplate(
        template=_PROMPT_TEMPLATE.format(num_functions=num_functions),
        input_variables=["context", "question"],
    )
    chain_type_kwargs = {"prompt": prompt}

    return RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, max_tokens=1000),
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": num_retrievals}
        ),
        chain_type_kwargs=chain_type_kwargs,
    )


_RESPONSE_FORMAT = """
### Function name: `{function}`

Function type: **{type}**

Description: **{description}**

Example:
```python3
{example}
```

#### Other possible matches:
{other_matches}

{additional_info}
"""

_DOC_STRING_FORMAT = """
### Function docstring:

```
{docstring}
```

"""


def format_response(
    data_match,
    other_matches: Optional[List[str]] = None,
    docstring: bool = False,
    markdown: bool = True,
) -> Union[str, Markdown]:
    """
    Formats the response to a question in markdown format.

    Parameters
    ----------
    data_match : dict
        A dictionary containing information about the matched function.
    other_matches : Optional[List[str]], optional
        A list of other possible matches for the question, by default None
    docstring : bool, optional
        A boolean indicating whether to include the function's docstring in the response, by default False
    markdown : bool, optional
        A boolean indicating whether to return the response as a Markdown object, by default False

    Returns
    -------
    Union[str, Markdown]
        A string or Markdown object containing the formatted response.
    """
    if other_matches:
        other_funcs = "\n".join([f"- {match}" for match in other_matches])
    else:
        other_funcs = "None"

    if docstring:
        additional_info = _DOC_STRING_FORMAT.format(docstring=data_match["docstring"])
    else:
        additional_info = ""

    formatted_result = _RESPONSE_FORMAT.format(
        function=data_match["function_name"],
        type=data_match["type"],
        description=data_match["description"],
        example=data_match["example"],
        additional_info=additional_info,
        other_matches=other_funcs,
    )
    return Markdown(formatted_result) if markdown else formatted_result
