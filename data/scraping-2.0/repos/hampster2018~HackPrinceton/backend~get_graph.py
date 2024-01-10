import json
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


class DisplayKeyword(BaseModel):
    """Display a keyword to the user that summarizes a list of keyphrases."""

    keyword_to_display: str = Field(
        ...,
        description="This is the keyword that best summarizes the user's keyphrases.",
    )


def get_summary(
    keyphrases: list[tuple[str, float, int]], previous_summaries: list[str]
):
    # this will get the summary for a chunk of text
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # set up chat memory
    keyphrase_list = []
    for keyphrase in keyphrases:
        keyphrase_list.append(keyphrase[0])

    # get the summary
    # summary = llm.invoke(messages).content
    runnable = create_structured_output_runnable(
        output_schema=DisplayKeyword,
        llm=llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a summarizer function. Take the user message and output a single keyword that summarizes the keyphrases the user provided, but be as specific as possible. Do not be general with your keywords. Do not say anything other than JUST the keyword. DO NOT output any words on the avoid list. DO NOT output 'database'.",
                ),
                ("human", "{input}"),
            ]
        ),
    )
    summary = runnable.invoke(
        {
            "input": f"What is one keyword that best summarizes this list of keyphrases? Also, I will give you a list of words to avoid. Here is the list of keyphrases: {json.dumps(keyphrase_list)}. Here is the list of words to avoid: {json.dumps(previous_summaries)}"
        }
    )

    return summary


def get_graph_with_summaries(textbook_path: str):
    # this will get the existing topics from the textbook and generate a summary node for each
    # group of topics

    # these summary nodes will then be connected and flattened into a G = (V, E) graph
    # also, don't forget to connect all the topics to their respective summary nodes
    topics = None
    with open(textbook_path, "r") as f:
        topics = json.loads(f.read())

    # the format of topics is a dictionary with two keys: "keyphrases" and "relations"

    # keyphrases is a list of tuples of the form (keyphrase, score, chunk_index)
    # relations is a list of lists of the form [chunk_index_1, chunk_index_2]

    # get the last chunk index
    last_chunk_index = topics["keyphrases"][-1][2]
    final_graph = {"nodes": [], "edges": []}

    for i in range(last_chunk_index + 1):
        # get all the keyphrases for this chunk
        keyphrases = []
        for keyphrase in topics["keyphrases"]:
            if keyphrase[2] == i:
                keyphrases.append(keyphrase)

        # get the previous summary nodes
        previous_summaries = []
        for node in final_graph["nodes"]:
            if node["group"] == i - 1:
                previous_summaries.append(node)

        summary = get_summary(keyphrases, previous_summaries).keyword_to_display

        # add the summary node the graph at its chunk index
        final_graph["nodes"].append(
            {"id": i, "label": summary.lower(), "group": i, "completed": False}
        )

    # add the rest of the nodes to the graph now that we have summary nodes at the right indices
    for i, keyphrase in enumerate(topics["keyphrases"]):
        final_graph["nodes"].append(
            {
                "id": i + last_chunk_index + 1,
                "label": keyphrase[0].lower(),
                "group": keyphrase[2],
                "completed": False,
            }
        )

        # add the edges connecting the summary nodes to the keyphrase nodes
        final_graph["edges"].append([keyphrase[2], i + last_chunk_index + 1])

    # for any summary nodes that are not sequentially connected, add an edge between them
    for i in range(last_chunk_index):
        final_graph["edges"].append([i, i + 1])

    final_graph["edges"].extend(topics["relations"])

    return final_graph


if __name__ == "__main__":
    get_graph_with_summaries("database_textbook")
