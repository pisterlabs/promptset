from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GraphQLAPIWrapper
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.schema.output_parser import OutputParserException


from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streamlit.mutable_expander import MutableExpander
from langchain.schema import AgentAction, AgentFinish, LLMResult


st.set_page_config(
    page_title="Ancient Language Librarian",
    page_icon="🏛️",
    # layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------Hide Streamlit footer----------------------------
footer = """

<style>footer {
	
	visibility: hidden;
	
	}<style>
<div class='footer'>
<p>Please click to send <a style='display:block;text-align:center;' 
href='mailto:ryder.wishart@clear.bible' target='_blank'>feedback</a></p>
</div>"""

st.markdown(footer, unsafe_allow_html=True)
# --------------------------------------------------------------------

runs_dir = Path(__file__).parent / "runs"
runs_dir.mkdir(exist_ok=True)

# ## Set up MACULA dataframes


@st.cache_data
def load_dataframes():
    verse_df = pd.read_csv("databases/preprocessed-macula-dataframes/verse_df.csv")
    mg = pd.read_csv("databases/preprocessed-macula-dataframes/mg.csv")
    # mh = pd.read_csv("preprocessed-macula-dataframes/mh.csv")
    return verse_df, mg  # , mh


verse_df, mg = load_dataframes()

from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool, tool

# from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
import pickle
import time
from typing import Any, TypedDict
from langchain.callbacks.base import BaseCallbackHandler


# This is intentionally not an enum so that we avoid serializing a
# custom class with pickle.
class CallbackType:
    ON_LLM_START = "on_llm_start"
    ON_LLM_NEW_TOKEN = "on_llm_new_token"
    ON_LLM_END = "on_llm_end"
    ON_LLM_ERROR = "on_llm_error"
    ON_TOOL_START = "on_tool_start"
    ON_TOOL_END = "on_tool_end"
    ON_TOOL_ERROR = "on_tool_error"
    ON_TEXT = "on_text"
    ON_CHAIN_START = "on_chain_start"
    ON_CHAIN_END = "on_chain_end"
    ON_CHAIN_ERROR = "on_chain_error"
    ON_AGENT_ACTION = "on_agent_action"
    ON_AGENT_FINISH = "on_agent_finish"


# We use TypedDict, rather than NamedTuple, so that we avoid serializing a
# custom class with pickle. All of this class's members should be basic Python types.
class CallbackRecord(TypedDict):
    callback_type: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    time_delta: float  # Number of seconds between this record and the previous one


def load_records_from_file(path: str) -> list[CallbackRecord]:
    """Load the list of CallbackRecords from a pickle file at the given path."""
    with open(path, "rb") as file:
        records = pickle.load(file)

    if not isinstance(records, list):
        raise RuntimeError(f"Bad CallbackRecord data in {path}")
    return records


def playback_callbacks(
    handlers: list[BaseCallbackHandler],
    records_or_filename: list[CallbackRecord] | str,
    max_pause_time: float,
) -> str:
    if isinstance(records_or_filename, list):
        records = records_or_filename
    else:
        records = load_records_from_file(records_or_filename)

    for record in records:
        pause_time = min(record["time_delta"] / 2, max_pause_time)
        if pause_time > 0:
            time.sleep(pause_time)

        for handler in handlers:
            if record["callback_type"] == CallbackType.ON_LLM_START:
                handler.on_llm_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_NEW_TOKEN:
                handler.on_llm_new_token(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_END:
                handler.on_llm_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_LLM_ERROR:
                handler.on_llm_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_START:
                handler.on_tool_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_END:
                handler.on_tool_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TOOL_ERROR:
                handler.on_tool_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_TEXT:
                handler.on_text(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_START:
                handler.on_chain_start(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_END:
                handler.on_chain_end(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_CHAIN_ERROR:
                handler.on_chain_error(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_AGENT_ACTION:
                handler.on_agent_action(*record["args"], **record["kwargs"])
            elif record["callback_type"] == CallbackType.ON_AGENT_FINISH:
                handler.on_agent_finish(*record["args"], **record["kwargs"])

    # Return the agent's result
    for record in records:
        if record["callback_type"] == CallbackType.ON_AGENT_FINISH:
            return record["args"][0][0]["output"]

    return "[Missing Agent Result]"


class CapturingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._records: list[CallbackRecord] = []
        self._last_time: float | None = None

    def dump_records_to_file(self, path: str) -> None:
        """Write the list of CallbackRecords to a pickle file at the given path."""
        with open(path, "wb") as file:
            pickle.dump(self._records, file)

    def _append_record(
        self, type: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        time_now = time.time()
        time_delta = time_now - self._last_time if self._last_time is not None else 0
        self._last_time = time_now
        self._records.append(
            CallbackRecord(
                callback_type=type, args=args, kwargs=kwargs, time_delta=time_delta
            )
        )

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_START, args, kwargs)

    def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_NEW_TOKEN, args, kwargs)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_END, args, kwargs)

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_LLM_ERROR, args, kwargs)

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_START, args, kwargs)

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_END, args, kwargs)

    def on_tool_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TOOL_ERROR, args, kwargs)

    def on_text(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_TEXT, args, kwargs)

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_START, args, kwargs)

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_END, args, kwargs)

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_CHAIN_ERROR, args, kwargs)

    def on_agent_action(self, *args: Any, **kwargs: Any) -> Any:
        self._append_record(CallbackType.ON_AGENT_ACTION, args, kwargs)

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        self._append_record(CallbackType.ON_AGENT_FINISH, args, kwargs)


# A hack to "clear" the previous result when submitting a new prompt. This avoids
# the "previous run's text is grayed-out but visible during rerun" Streamlit behavior.
class DirtyState:
    NOT_DIRTY = "NOT_DIRTY"
    DIRTY = "DIRTY"
    UNHANDLED_SUBMIT = "UNHANDLED_SUBMIT"


def get_dirty_state() -> str:
    return st.session_state.get("dirty_state", DirtyState.NOT_DIRTY)


def set_dirty_state(state: str) -> None:
    st.session_state["dirty_state"] = state


def with_clear_container(submit_clicked: bool) -> bool:
    if get_dirty_state() == DirtyState.DIRTY:
        if submit_clicked:
            set_dirty_state(DirtyState.UNHANDLED_SUBMIT)
            st.experimental_rerun()
        else:
            set_dirty_state(DirtyState.NOT_DIRTY)

    if submit_clicked or get_dirty_state() == DirtyState.UNHANDLED_SUBMIT:
        set_dirty_state(DirtyState.DIRTY)
        return True

    return False


# # Expand functionality for more tools using DB lookups

# from langchain.chains.question_answering import load_qa_chain
# from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# from langchain.embeddings import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings()

# bible_persist_directory = "/Users/ryderwishart/genesis/databases/berean-bible-database"
# bible_chroma = Chroma(
#     "berean-bible", embeddings, persist_directory=bible_persist_directory
# )
# print(bible_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

# encyclopedic_persist_directory = "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/encyclopedic"
# encyclopedic_chroma = Chroma(
#     persist_directory=encyclopedic_persist_directory,
#     embedding_function=embeddings,
#     collection_name="encyclopedic",
# )
# print(
#     encyclopedic_chroma.similarity_search_with_score(
#         "What is a sarcophagus?", search_type="similarity", k=1
#     )
# )

# theology_persist_directory = (
#     "/Users/ryderwishart/biblical-machine-learning/gpt-inferences/databases/theology"
# )
# theology_chroma = Chroma(
#     "theology", embeddings, persist_directory=theology_persist_directory
# )
# print(theology_chroma.search("jesus speaks to peter", search_type="similarity", k=1))

# # persist_directory = '/Users/ryderwishart/genesis/databases/itemized-prose-contexts copy' # NOTE: Itemized prose contexts are in this db
# persist_directory = '/Users/ryderwishart/genesis/databases/prose-contexts' # NOTE: Full prose contexts are in this db
# context_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="prosaic_contexts_itemized")
# print(context_chroma.similarity_search_with_score('jesus (s) speaks (v) to peter (o)', search_type='similarity', k=1))

# persist_directory = (
#     "/Users/ryderwishart/genesis/databases/prose-contexts-shorter-itemized"
# )
# context_chroma = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embeddings,
#     collection_name="prosaic_contexts_shorter_itemized",
# )

SAVED_SESSIONS = {}
# Populate saved sessions from runs_dir
for path in runs_dir.glob("*.pickle"):
    with open(path, "rb") as f:
        SAVED_SESSIONS[path.stem] = path


"# 🏛️📚 Ancient Language Librarian"
"Ask questions about the Bible. Open sidebar to enter your own API key and ask new questions."
"**This is a work in progress.** Please share issues with [ryderwishart at gmail dot com]"
"*Note: there will be a delay when the agent calls certain external APIs that need to wake up. Subsequent calls should be quicker.*"

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)

# Add a radio button selector for the model
model_name = st.sidebar.radio(
    "Select Model",
    ("gpt-3.5-turbo", "gpt-4"),
    help="Select the model to use for the chat.",
)

# Improve the linguistic data lookup tool with discourse feature definitions
discourse_types = {
    "Main clauses": {
        "description": "Main clauses are the top-level clauses in a sentence. They are the clauses that are not embedded in other clauses."
    },
    "Historical Perfect": {
        "description": "Highlights not the speech or act to which it refers but the event(s) that follow (DFNTG §12.2)."
    },
    "Specific Circumstance": {
        "description": "The function of ἐγενετο ‘it came about’ and an immediately following temporal expression varies with the author (see DFNTG §10.3). In Matthew’s Gospel, it usually marks major divisions in the book (e.g. Mt 7:28). In Luke-Acts, in contrast, ‘it picks out from the general background the specific circumstance for the foreground events that are to follow’ (ibid.), as in Acts 9:37 (see also Mt 9:10)."
    },
    "Verb Focus+": {
        "description": "Verb in final position in clause demonstrates verb focus."
    },
    "Articular Pronoun": {
        "description": "Articular pronoun, which often introduces an ‘intermediate step’ in a reported conversation."
    },
    "Topical Genitive": {
        "description": "A genitival constituent that is nominal is preposed within the noun phrase for two purposes: 1) to bring it into focus; 2) within a point of departure, to indicate that it is the genitive in particular which relates to a corresponding constituent of the context.(DFNTG §4.5)"
    },
    "Embedded DFE": {
        "description": "'Dominant focal elements' embedded within a constituent in P1."
    },
    "Reported Speech": {"description": "Reported speech."},
    "Ambiguous": {"description": "Marked but ambiguous constituent order."},
    "Over-encoding": {
        "description": "Any instance in which more encoding than the default is employed to refer to an active participant or prop. Over-encoding is used in Greek, as in other languages: to mark the beginning of a narrative unit (e.g. Mt 4:5); and to highlight the action or speech concerned (e.g. Mt 4:7)."
    },
    "Highlighter": {
        "description": "Presentatives - Interjections such as ἰδού and ἴδε ‘look!, see!’ typically highlight what immediately follows (Narr §5.4.2, NonNarr §7.7.3)."
    },
    "Referential PoD": {
        "description": "Pre-verbal topical subject other referential point of departure (NARR §3.1, NonNarr §4.3, DFNTG §§2.2, 2.8; as in 1 Th 1:6)."
    },
    "annotations": {"description": "Inline annotations."},
    "Left-Dislocation": {
        "description": "Point of departure - A type of SENTENCE in which one of the CONSTITUENTS appears in INITIAL position and its CANONICAL position is filled by a PRONOUN or a full LEXICAL NOUN PHRASE with the same REFERENCE, e.g. John, I like him/the old chap.”"
    },
    "Focus+": {
        "description": "Constituents placed in P2 to give them focal prominence."
    },
    "Tail-Head linkage": {
        "description": "Point of departure involving renewal - Tail-head linkage involves “the repetition in a subordinate clause, at the beginning (the ‘head’) of a new sentence, of at least the main verb of the previous sentence (the tail)” (Dooley & Levinsohn 2001:16)."
    },
    "Postposed them subject": {
        "description": "When a subject is postposed to the end of its clause (following nominals or adjuncts), it is marked ThS+ (e.g. Lk 1:41 [twice]). Such postposing typically marks as salient the participant who performs the next event in chronological sequence in the story (see Levinsohn 2014)."
    },
    "EmbeddedRepSpeech": {
        "description": "Embedded reported speech - speech that is reported within a reported speech."
    },
    "Futuristic Present": {
        "description": "Highlights not the speech or act to which it refers but the event(s) that follow (DFNTG §12.2)."
    },
    "OT quotes": {"description": "Old Testament quotations."},
    "Constituent Negation": {
        "description": "Negative pro-forms when they are in P2 indicate that the constituent has been negated rather than the clause as a whole."
    },
    "Split Focal": {
        "description": "The second part of a focal constituent with only the first part in P2 (NonNarr §5.5, DFNTG §4.4)."
    },
    "Right-Dislocated": {
        "description": "Point of departure - A type of SENTENCE in which one of the CONSTITUENTS appears in FINAL position and its CANONICAL position is filled by a PRONOUN with the same REFERENCE, e.g. ... He’s always late, that chap."
    },
    "Appositive": {"description": "Appositive"},
    "Situational PoD": {
        "description": "Situational point of departure (e.g. temporal, spatial, conditional―(NARR §3.1, NonNarr §4.3, DFNTG §§2.2, 2.8; as in 1 Th 3:4)."
    },
    "Historical Present": {
        "description": "Highlights not the speech or act to which it refers but the event(s) that follow (DFNTG §12.2)."
    },
    "Noun Incorporation": {
        "description": "Some nominal objects that appear to be in P2 may precede their verb because they have been “incorporated” (Rosen 1989) in the verb phrase. Typically, the phrase consists of an indefinite noun and a “light verb” such as “do, give, have, make, take” (Wikipedia entry on Light Verbs)."
    },
    "Thematic Prominence": {
        "description": "Thematic prominence - In Greek, prominence is given to active participants and props who are the current centre of attention (NARR §4.6) by omitting the article (DFNTG §§9.2.3-9.4), by adding αυτος ‘-self’ (e.g. in 1 Th 3:11), by using the proximal demonstrative οὗτος (NARR chap. 9, Appendix 1; e.g. in 3:3), and by postposing the constituent concerned (e.g. Mt 14:29). If such constituents are NOT in postion P1, they are demonstrating topical prominence."
    },
    "Cataphoric Focus": {
        "description": "An expression that points forward to and highlights something which ‘is about to be expressed.’"
    },
    "Cataphoric referent": {
        "description": "The clause or sentence to which a cataphoric reference refers when NOT introduced with ὅτι or ἵνα."
    },
    "DFE": {
        "description": "Constituents that may be moved from their default position to the end of a proposition to give them focal prominence include verbs, pronominals and objects that follow adjuncts (NonNarr §5.3, DFNTG §3.5). Such constituents, also called ‘dominant focal elements’or DFEs (Heimedinger 1999:167)."
    },
    "Embedded Focus+": {
        "description": "A constituent of a phrase or embedded clause preposed for focal prominence."
    },
}


@tool  # FIXME: use atlas agent instead
def linguistic_data_lookup_tool(query):
    """Query the linguistic data for relevant documents and add explanatory suffix if appropriate."""
    context_docs = context_chroma.similarity_search(query, k=3)
    explanatory_suffix = "Here are the definitions of the relevant discourse features:"
    include_suffix_flag = False
    for discourse_type in discourse_types.keys():
        if discourse_type in query:
            explanatory_suffix += f"\n\n{discourse_type}: {discourse_types[discourse_type]['description']}"
            include_suffix_flag = True
    if include_suffix_flag:
        context_docs.append(explanatory_suffix)
    return str(context_docs)


@tool
def query_bible(query: str):
    """Ask a question of the Berean Bible endpoint."""
    endpoint = "https://ryderwishart--bible-chroma-get-documents.modal.run/"
    url_encoded_query = query.replace(" ", "%20")
    url = f"{endpoint}?query={url_encoded_query}"

    try:
        response = requests.get(url)
        return response.json()
    except:
        return {
            "error": "There was an error with the request. Please reformat request or try another tool."
        }


# query encyclopedic data using https://ryderwishart--tyndale-chroma-get-documents.modal.run/?query=jesus%20speaks%20to%20john
@tool
def query_encyclopedia(query: str):
    """Ask a question of the Tyndale Encyclopedia endpoint."""
    endpoint = "https://ryderwishart--tyndale-chroma-get-documents.modal.run/"
    url_encoded_query = query.replace(" ", "%20")
    url = f"{endpoint}?query={url_encoded_query}"

    try:
        response = requests.get(url)
        return response.json()
    except:
        return {
            "error": "There was an error with the request. Please reformat request or try another tool."
        }


atlas_endpoint = "https://macula-atlas-api-qa-25c5xl4maa-uk.a.run.app/graphql/"


def get_macula_atlas_schema():
    """Query the macula atlas api for its schema"""
    global atlas_endpoint
    query = """
    query IntrospectionQuery {
        __schema {
            types {
                name
                kind
                fields {
                    name
                    type {
                        name
                        kind
                        ofType {
                            name
                            kind
                        }
                    }
                }
            }
        }
    }"""
    request = requests.post(atlas_endpoint, json={"query": query})
    json_output = request.json()

    # Simplify the schema
    simplified_schema = {}
    for type_info in json_output["data"]["__schema"]["types"]:
        if not type_info["name"].startswith("__"):
            fields = type_info.get("fields")
            if fields is not None and fields is not []:
                simplified_schema[type_info["name"]] = {
                    "kind": type_info["kind"],
                    "fields": ", ".join(
                        [
                            field["name"]
                            for field in fields
                            if not field["name"].startswith("__")
                        ]
                    ),
                }
            else:
                simplified_schema[type_info["name"]] = {
                    "kind": type_info["kind"],
                }

    return simplified_schema

    # Convert the simplified schema to YAML
    # yaml_output = yaml.dump(simplified_schema, default_flow_style=False)

    # return yaml_output


@tool
def answer_question_using_atlas(query: str, show_sources: bool = False):
    """Answer a question using the Macula Atlas API.

    Step 1. find the most relevant Bible verse reference using the Berean Bible endpoint
    Step 2. find the relevant discourse features using the Macula Atlas API
    Step 3. add explanatory note with glosses for found discourse features
    """

    global atlas_endpoint
    graphql_fields = (
        get_macula_atlas_schema()
    )  # Only call this when the ATLAS agent is called
    examples = """
    ## All features and instances for 2 Corinthians 8:2
    query AnnotationFeatures {
        annotationFeatures(filters: {reference: "2CO 8:2"}) {
        label
            uri
            instances(filters: {reference: "2CO 8:2"}) {
                uri
                tokens {
                    ref
                }
            }
        }
    }
    
    ## First 10 annotations with featureLabel "Main clauses"
    query Annotations {
        annotations(
            filters: { featureLabel: "Main clauses" }
            pagination: { limit: 10, offset: 0 }
        ) {
            uri
            depth
            tokens {
                ref
            }
        }
    }

    ## All features and instances for John 1:12
    query {
        annotationFeatures(filters: {reference: "JHN 1:12"}) {
            label
            uri
            instances(filters: {reference: "JHN 1:12"}) {
                uri
                tokens {
                    ref
                }
            }
        }
    }
    
    ## All features for John 3:16
    query AnnotationFeatures {
        annotationFeatures(filters: {reference: "JHN 3:16"}) {
        label
            uri
            data
        }
    }
    
    Note that the bible reference is repeated for features and for instances. If searching for features without a reference filter, be sure to use pagination to limit the number of results returned!
"""

    prompt = f"""Here are some example queries for the graphql endpoint described below:
    {examples}

    Answer the following question: {query} in the graphql database that has this schema {graphql_fields}"""

    try:
        result = atlas_agent.run(prompt)
    except Exception as e:
        prompt += f"\n\nThere was an error with the request.\nError: {e}\n\nPlease reformat GraphQL query (avoid issues with backticks if possible)."
        result = atlas_agent.run(prompt)

    # Check result for discourse features and add explanatory suffix if appropriate
    discourse_features_in_result = []
    for discourse_type in discourse_types.keys():
        if discourse_type in result:
            discourse_features_in_result.append(discourse_type)
    if len(discourse_features_in_result) > 0:
        explanatory_suffix = (
            "Here are the definitions of the relevant discourse features:"
        )
        for discourse_feature in discourse_features_in_result:
            explanatory_suffix += f"\n\n{discourse_feature}: {discourse_types[discourse_feature]['description']}"
        result += explanatory_suffix

    return result


@tool
def syntax_qa_chain(query):
    """Use langchain to complete QA chain for syntax question"""
    global llm
    prompt_template = """The contexts provided below follow a simple syntax markup, where 
    s=subject
    v=verb
    o=object
    io=indirect object
    +=adverbial modifier
    p=non-verbal predicate
    
    Answer each question by extracting the relevant syntax information from the provided context:
    Q: What is the subject of the main verb in Mark 1:15?
    Context: And (Καὶ)] 
[[+: after (μετὰ)] the (τὸ)] 
[[v: delivering up (παραδοθῆναι)] [s: - (τὸν)] of John (Ἰωάνην)] [v: came (ἦλθεν)] [s: - (ὁ)] Jesus (Ἰησοῦς)] [+: into (εἰς)] - (τὴν)] Galilee (Γαλιλαίαν)] [+: 
[[v: proclaiming (κηρύσσων)] [o: the (τὸ)] gospel (εὐαγγέλιον)] - (τοῦ)] of God (Θεοῦ)] and (καὶ)] 
[[v: saying (λέγων)] [+: - (ὅτι)] 
[[v: Has been fulfilled (Πεπλήρωται)] [s: the (ὁ)] time (καιρὸς)] and (καὶ)] 
[[v: has drawn near (ἤγγικεν)] [s: the (ἡ)] kingdom (βασιλεία)] - (τοῦ)] of God·(Θεοῦ)] 
[[v: repent (μετανοεῖτε)] and (καὶ)] 
[[v: believe (πιστεύετε)] [+: in (ἐν)] the (τῷ)] gospel.(εὐαγγελίῳ)]
    A: The subject of the main verb is Jesus ([s: - (ὁ)] Jesus (Ἰησοῦς)])
    
    Q: Who is the object of Jesus' command in Matthew 28:19?
    Context: therefore (οὖν)] 
[
[[+: [v: Having gone (πορευθέντες)] [v: disciple (μαθητεύσατε)] [o: all (πάντα)] the (τὰ)] nations,(ἔθνη)] 
[[+: [v: baptizing (βαπτίζοντες)] [o: them (αὐτοὺς)] [+: in (εἰς)] the (τὸ)] name (ὄνομα)] of the (τοῦ)] Father (Πατρὸς)] and (καὶ)] of the (τοῦ)] Son (Υἱοῦ)] and (καὶ)] of the (τοῦ)] Holy (Ἁγίου)] Spirit,(Πνεύματος)] 
[[+: [v: teaching (διδάσκοντες)] 
[[o: [s: them (αὐτοὺς)] [v: to observe (τηρεῖν)] [o: all things (πάντα)] 
[[apposition: [o: whatever (ὅσα)] [v: I commanded (ἐνετειλάμην)] [io: you·(ὑμῖν)]
    A: In the verse, he commanded 'you' ([io: you·(ὑμῖν)])
    
    Q: What are the circumstances of the main clause in Luke 15:20?
    Context: And (καὶ)] 
[
[[+: [v: having risen up (ἀναστὰς)] [v: he went (ἦλθεν)] [+: to (πρὸς)] the (τὸν)] father (πατέρα)] of himself.(ἑαυτοῦ)] now (δὲ)] 
[[+: Still (ἔτι)] [s: he (αὐτοῦ)] [+: far (μακρὰν)] [v: being distant (ἀπέχοντος)] 
[[v: saw (εἶδεν)] [o: him (αὐτὸν)] [s: the (ὁ)] father (πατὴρ)] of him (αὐτοῦ)] and (καὶ)] 
[[v: was moved with compassion,(ἐσπλαγχνίσθη)] and (καὶ)] 
[
[[+: [v: having run (δραμὼν)] [v: fell (ἐπέπεσεν)] [+: upon (ἐπὶ)] the (τὸν)] neck (τράχηλον)] of him (αὐτοῦ)] and (καὶ)] 
[[v: kissed (κατεφίλησεν)] [o: him.(αὐτόν)]
    A: The implied subject goes 'to his own father' ([+: to (πρὸς)] the (τὸν)] father (πατέρα)] of himself.(ἑαυτοῦ)])
    
    Q: What does Jesus tell his disciples to do in Matthew 5:44 regarding their enemies, and what is the reason he gives for this command?
    Context: however (δὲ)] 
[[s: I (ἐγὼ)] [v: say (λέγω)] [io: to you,(ὑμῖν)] [o: 
[[v: love (ἀγαπᾶτε)] [o: the (τοὺς)] enemies (ἐχθροὺς)] of you (ὑμῶν)]
    A: Jesus tells his disciples to love their enemies ([[v: love (ἀγαπᾶτε)] [o: the (τοὺς)] enemies (ἐχθροὺς)] of you (ὑμῶν)])
    
    Q: {question}
    Context: {context}
    A: """

    # llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"],
        ),
    )

    syntax_brackets_endpoint = (
        "https://ryderwishart--syntax-agent-get-syntax-for-query.modal.run/?query="
    )
    context = requests.get(syntax_brackets_endpoint + query).json()

    # return {
    #     "answer": llm_chain.predict(context=context, question=query),
    #     "context": context,
    # }

    return llm_chain.predict(context=context, question=query)


tools = []

if user_openai_api_key:
    openai_api_key = user_openai_api_key
    enable_custom = True
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        # model_name="gpt-3.5-turbo-16k",
        model_name=model_name,
        temperature=0,
        streaming=True,
    )

    from langchain.agents import create_pandas_dataframe_agent

    macula_greek_verse_agent = create_pandas_dataframe_agent(
        llm,
        # mg, # verse_df (?)
        verse_df,
        # verbose=True,
    )

    macula_greek_words_agent = create_pandas_dataframe_agent(
        llm,
        # mg, # verse_df (?)
        mg,
        # verbose=True,
    )

    atlas_tools = load_tools(
        ["graphql"],
        graphql_endpoint=atlas_endpoint,
        llm=llm,
    )
    atlas_agent = initialize_agent(
        atlas_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    tools = [
        Tool(
            name="Bible Verse Reader Lookup",
            func=query_bible.run,
            description="useful for finding verses that are similar to the user's query; not suitable for complex queries. Be very careful to check whether the verses are actually relevant to the user's question and not just similar to the user's question in superficial ways. Input should be description related to the user's query using relevant keywords. Do NOT input a verse range, as this will not work.",
            # return_direct=True,
        ),
        Tool(
            name="Bible Words Lookup",
            func=macula_greek_words_agent.run,  # Note: using the NT-only agent here
            description="useful for finding information about individual biblical words from a Greek words dataframe, which includes glosses, lemmas, normalized forms, and more. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the words themselves. Input should be a fully formed question related to the user's query.",
            return_direct=True,
        ),
        Tool(
            name="Bible Verse Dataframe Tool",
            func=macula_greek_verse_agent.run,  # Note: using the NT-only agent here
            description="useful for finding information about Bible verses in a bible verse dataframe in case counting, grouping, aggregating, or list building is required. This tool is not useful for grammar and syntax questions (about subjects, objects, verbs, etc.), but is useful for finding information about the verses (English or Greek or Greek lemmas) themselves. Input should be a fully formed question related to the user's query.",
            return_direct=True,
        ),
        Tool(
            name="Linguistic Data Lookup",
            # func=linguistic_data_lookup_tool.run,
            func=answer_question_using_atlas.run,
            description="useful for finding answers about linguistics, discourse, situational context, participants, semantic roles (source/agent, process, goal, etc.), or who the speakers are in a passage. Input should be a verse reference only.",
            return_direct=True,
        ),
        Tool(
            name="Syntax Data Lookup",
            func=syntax_qa_chain.run,
            description="useful for finding syntax data about the user's query. Use this if the user is asking a question that relates to a sentence's structure, such as 'who is the subject of this sentence?' or 'what are the circumstances of this verb?'. Input should be a fully formed question.",
            return_direct=True,
        ),
        # Tool(
        #     name="Theological Data Lookup",
        #     func=lambda x: theology_chroma.search(x, search_type="similarity", k=5),
        #     description="if you can't find a linguistic answer, this is useful only for finding theological data about the user's query. Use this if the user is asking about theological concepts or value-oriented questions about 'why' the Bible says certain things. Always be sure to cite the source of the data. Input should be a fully formed question.",
        # ),
        Tool(
            name="Encyclopedic Data Lookup",
            # func=lambda x: encyclopedic_chroma.similarity_search(x, k=5),
            func=query_encyclopedia.run,
            description="useful for finding encyclopedic data about the user's query. Use this if the user is asking about historical, cultural, geographical, archaeological, theological, or other types of information from secondary sources. Input should be a fully formed question. If the question is theological/philosophical, ALWAYS be sure to note that 'some sources say...' and 'be sure to check the Bible itself to confirm these claims.'",
        ),
        # Tool(
        #     name="Any Other Kind of Question Tool",
        #     func=lambda x: "Sorry, I don't know!",
        #     description="This tool is for vague, broad, ambiguous questions. Input should be a fully formed question.",
        # ),
    ]
    # function_llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4-0613")

else:
    openai_api_key = "not_supplied"
    enable_custom = False
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        # model_name="gpt-3.5-turbo-16k",
        model_name=model_name,
        temperature=0,
        streaming=True,
    )
    # function_llm = ChatOpenAI(openai_api_key=openai_api_key)


# Initialize agent
mrkl = initialize_agent(
    tools,
    # function_llm,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

with st.form(key="form"):
    if not enable_custom:
        "Ask one of the sample questions, or enter your API Key in the sidebar to ask your own custom questions."
    prefilled = (
        st.selectbox(
            "Sample questions",
            sorted([key.replace("_", " ") for key in SAVED_SESSIONS.keys()]),
        )
        or ""
    )
    user_input = ""

    if enable_custom:
        user_input = st.text_input("Or, ask your own question")
    if not user_input:
        user_input = prefilled
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()

# st.write(SAVED_SESSIONS)


#################################
# Customized Streamlit Callback #
#################################


# if TYPE_CHECKING:
from streamlit.delta_generator import DeltaGenerator


def _convert_newlines(text: str) -> str:
    """Convert newline characters to markdown newline sequences
    (space, space, newline).
    """
    return text.replace("\n", "  \n")


CHECKMARK_EMOJI = "✅"
THINKING_EMOJI = ":thinking_face:"
HISTORY_EMOJI = ":books:"
EXCEPTION_EMOJI = "⚠️"


class LLMThoughtState(Enum):
    # The LLM is thinking about what to do next. We don't know which tool we'll run.
    THINKING = "THINKING"
    # The LLM has decided to run a tool. We don't have results from the tool yet.
    RUNNING_TOOL = "RUNNING_TOOL"
    # We have results from the tool.
    COMPLETE = "COMPLETE"


class ToolRecord(NamedTuple):
    name: str
    input_str: str


class LLMThoughtLabeler:
    """
    Generates markdown labels for LLMThought containers. Pass a custom
    subclass of this to StreamlitCallbackHandler to override its default
    labeling logic.
    """

    def get_initial_label(self) -> str:
        """Return the markdown label for a new LLMThought that doesn't have
        an associated tool yet.
        """
        return f"{THINKING_EMOJI} **Thinking...**"

    def get_tool_label(self, tool: ToolRecord, is_complete: bool) -> str:
        """Return the label for an LLMThought that has an associated
        tool.

        Parameters
        ----------
        tool
            The tool's ToolRecord

        is_complete
            True if the thought is complete; False if the thought
            is still receiving input.

        Returns
        -------
        The markdown label for the thought's container.

        """
        input = tool.input_str
        name = tool.name
        emoji = CHECKMARK_EMOJI if is_complete else THINKING_EMOJI
        if name == "_Exception":
            emoji = EXCEPTION_EMOJI
            name = "Parsing error"
        idx = min([60, len(input)])
        input = input[0:idx]
        if len(tool.input_str) > idx:
            input = input + "..."
        input = input.replace("\n", " ")
        label = f"{emoji} **{name}:** {input}"
        return label

    def get_history_label(self) -> str:
        """Return a markdown label for the special 'history' container
        that contains overflow thoughts.
        """
        return f"{HISTORY_EMOJI} **History**"

    def get_final_agent_thought_label(self) -> str:
        """Return the markdown label for the agent's final thought -
        the "Now I have the answer" thought, that doesn't involve
        a tool.
        """
        return f"{CHECKMARK_EMOJI} **Complete!**"


class LLMThought:
    def __init__(
        self,
        parent_container: DeltaGenerator,
        labeler: LLMThoughtLabeler,
        expanded: bool,
        collapse_on_complete: bool,
    ):
        self._container = MutableExpander(
            parent_container=parent_container,
            label=labeler.get_initial_label(),
            expanded=expanded,
        )
        self._state = LLMThoughtState.THINKING
        self._llm_token_stream = ""
        self._llm_token_writer_idx: Optional[int] = None
        self._last_tool: Optional[ToolRecord] = None
        self._collapse_on_complete = collapse_on_complete
        self._labeler = labeler

    @property
    def container(self) -> MutableExpander:
        """The container we're writing into."""
        return self._container

    @property
    def last_tool(self) -> Optional[ToolRecord]:
        """The last tool executed by this thought"""
        return self._last_tool

    def _reset_llm_token_stream(self) -> None:
        self._llm_token_stream = ""
        self._llm_token_writer_idx = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str]) -> None:
        self._reset_llm_token_stream()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # This is only called when the LLM is initialized with `streaming=True`
        self._llm_token_stream += _convert_newlines(token)
        self._llm_token_writer_idx = self._container.markdown(
            self._llm_token_stream, index=self._llm_token_writer_idx
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # `response` is the concatenation of all the tokens received by the LLM.
        # If we're receiving streaming tokens from `on_llm_new_token`, this response
        # data is redundant
        self._reset_llm_token_stream()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._container.markdown("**LLM encountered an error...**")
        self._container.exception(error)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        # Called with the name of the tool we're about to run (in `serialized[name]`),
        # and its input. We change our container's label to be the tool name.
        self._state = LLMThoughtState.RUNNING_TOOL
        tool_name = serialized["name"]
        self._last_tool = ToolRecord(name=tool_name, input_str=input_str)
        self._container.update(
            new_label=self._labeler.get_tool_label(self._last_tool, is_complete=False)
        )

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._container.markdown(f"**{output}**")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._container.markdown("**Tool encountered an error...**")
        self._container.exception(error)

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        # Called when we're about to kick off a new tool. The `action` data
        # tells us the tool we're about to use, and the input we'll give it.
        # We don't output anything here, because we'll receive this same data
        # when `on_tool_start` is called immediately after.
        pass

    def complete(self, final_label: Optional[str] = None) -> None:
        """Finish the thought."""
        if final_label is None and self._state == LLMThoughtState.RUNNING_TOOL:
            assert (
                self._last_tool is not None
            ), "_last_tool should never be null when _state == RUNNING_TOOL"
            final_label = self._labeler.get_tool_label(
                self._last_tool, is_complete=True
            )
        self._state = LLMThoughtState.COMPLETE
        if self._collapse_on_complete:
            self._container.update(new_label=final_label, new_expanded=False)
        else:
            self._container.update(new_label=final_label)

    def clear(self) -> None:
        """Remove the thought from the screen. A cleared thought can't be reused."""
        self._container.clear()


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        *,
        max_thought_containers: int = 4,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        thought_labeler: Optional[LLMThoughtLabeler] = None,
    ):
        """Create a StreamlitCallbackHandler instance.

        Parameters
        ----------
        parent_container
            The `st.container` that will contain all the Streamlit elements that the
            Handler creates.
        max_thought_containers
            The max number of completed LLM thought containers to show at once. When
            this threshold is reached, a new thought will cause the oldest thoughts to
            be collapsed into a "History" expander. Defaults to 4.
        expand_new_thoughts
            Each LLM "thought" gets its own `st.expander`. This param controls whether
            that expander is expanded by default. Defaults to True.
        collapse_completed_thoughts
            If True, LLM thought expanders will be collapsed when completed.
            Defaults to True.
        thought_labeler
            An optional custom LLMThoughtLabeler instance. If unspecified, the handler
            will use the default thought labeling logic. Defaults to None.
        """
        self._parent_container = parent_container
        self._history_parent = parent_container.container()
        self._history_container: Optional[MutableExpander] = None
        self._current_thought: Optional[LLMThought] = None
        self._completed_thoughts: List[LLMThought] = []
        self._max_thought_containers = max(max_thought_containers, 1)
        self._expand_new_thoughts = expand_new_thoughts
        self._collapse_completed_thoughts = collapse_completed_thoughts
        self._thought_labeler = thought_labeler or LLMThoughtLabeler()

    def _require_current_thought(self) -> LLMThought:
        """Return our current LLMThought. Raise an error if we have no current
        thought.
        """

        if self._current_thought is None:
            # print(
            #     "Current LLMThought is unexpectedly None!",
            #     "Creating new thought from parent container...",
            # )
            self._current_thought = LLMThought(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )
            # raise RuntimeError("Current LLMThought is unexpectedly None!")
        return self._current_thought

    def _get_last_completed_thought(self) -> Optional[LLMThought]:
        """Return our most recent completed LLMThought, or None if we don't have one."""
        if len(self._completed_thoughts) > 0:
            return self._completed_thoughts[len(self._completed_thoughts) - 1]
        return None

    @property
    def _num_thought_containers(self) -> int:
        """The number of 'thought containers' we're currently showing: the
        number of completed thought containers, the history container (if it exists),
        and the current thought container (if it exists).
        """
        count = len(self._completed_thoughts)
        if self._history_container is not None:
            count += 1
        if self._current_thought is not None:
            count += 1
        return count

    def _complete_current_thought(self, final_label: Optional[str] = None) -> None:
        """Complete the current thought, optionally assigning it a new label.
        Add it to our _completed_thoughts list.
        """
        thought = self._require_current_thought()
        thought.complete(final_label)
        self._completed_thoughts.append(thought)
        self._current_thought = None

    def _prune_old_thought_containers(self) -> None:
        """If we have too many thoughts onscreen, move older thoughts to the
        'history container.'
        """
        while (
            self._num_thought_containers > self._max_thought_containers
            and len(self._completed_thoughts) > 0
        ):
            # Create our history container if it doesn't exist, and if
            # max_thought_containers is > 1. (if max_thought_containers is 1, we don't
            # have room to show history.)
            if self._history_container is None and self._max_thought_containers > 1:
                self._history_container = MutableExpander(
                    self._history_parent,
                    label=self._thought_labeler.get_history_label(),
                    expanded=False,
                )

            oldest_thought = self._completed_thoughts.pop(0)
            if self._history_container is not None:
                self._history_container.markdown(oldest_thought.container.label)
                self._history_container.append_copy(oldest_thought.container)
            oldest_thought.clear()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        if self._current_thought is None:
            self._current_thought = LLMThought(
                parent_container=self._parent_container,
                expanded=self._expand_new_thoughts,
                collapse_on_complete=self._collapse_completed_thoughts,
                labeler=self._thought_labeler,
            )

        self._current_thought.on_llm_start(serialized, prompts)

        # We don't prune_old_thought_containers here, because our container won't
        # be visible until it has a child.

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_new_token(token, **kwargs)
        self._prune_old_thought_containers()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_end(response, **kwargs)
        self._prune_old_thought_containers()

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._require_current_thought().on_llm_error(error, **kwargs)
        self._prune_old_thought_containers()

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._require_current_thought().on_tool_start(serialized, input_str, **kwargs)
        self._prune_old_thought_containers()

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._require_current_thought().on_tool_end(
            output, color, observation_prefix, llm_prefix, **kwargs
        )
        self._complete_current_thought()

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self._require_current_thought().on_tool_error(error, **kwargs)
        self._prune_old_thought_containers()

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        self._require_current_thought().on_agent_action(action, color, **kwargs)
        self._prune_old_thought_containers()

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        if self._current_thought is not None:
            self._current_thought.complete(
                self._thought_labeler.get_final_agent_thought_label()
            )
            self._current_thought = None


##################################
# End Streamlit Callback Handler #
##################################


if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🏛️")
    st_callback = StreamlitCallbackHandler(answer_container)

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    path_user_input = "_".join(user_input.split(" "))

    # st.write(f"Checking if {path_user_input} is in {SAVED_SESSIONS.keys()}")

    if path_user_input in SAVED_SESSIONS.keys():
        print(f"Playing saved session: {user_input}")
        session_name = SAVED_SESSIONS[path_user_input]
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks([st_callback], str(session_path), max_pause_time=1)
    else:
        print(f"Running LangChain: {user_input} because not in SAVED_SESSIONS")
        capturing_callback = CapturingCallbackHandler()
        try:
            answer = mrkl.run(user_input, callbacks=[st_callback, capturing_callback])
        except OutputParserException as e:
            answer = e.args[0].split("LLM output: ")[1]
        pickle_filename = user_input.replace(" ", "_") + ".pickle"
        capturing_callback.dump_records_to_file(runs_dir / pickle_filename)

    answer_container.write(answer)


import base64

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_jpeg_as_page_bg(jpeg_file):
    bin_str = get_base64_of_bin_file(jpeg_file)
    page_bg_img = '''
    <style>
    body::after {
        content:"";
        background: url("data:image/jpeg;base64,''' + bin_str + '''");
        background-size: cover;
        background-position: 80%;
        opacity: 1;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        position: absolute;
        z-index: 1;   
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 1rem;
        margin-top: 4rem;
        padding-top: 0.5rem;
        z-index: 2;
    }
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_jpeg_as_page_bg('./assets/librarian.jpg')

