# create OpenAIAssistantAgent
from pydantic import BaseModel, Field# define pydantic model for auto-retrieval function
from typing import Tuple, List
from llama_index.tools import FunctionTool
from llama_index.agent import OpenAIAssistantAgent
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores import SupabaseVectorStore

from llama_index.tools import QueryEngineTool, ToolMetadata
import os
from dotenv import load_dotenv
load_dotenv()

vector_store = SupabaseVectorStore(
    postgres_connection_string=(
        f"postgresql://postgres:{os.getenv('VECTOR_DATABASE_PW')}@db.rgvrtfssleyejerbzqbv.supabase.co:5432/postgres"
    ),
    collection_name="research_papers",
)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# laod index from supabase
index = VectorStoreIndex.from_vector_store(vector_store)


c_elegans_tool = QueryEngineTool(
    query_engine=index.as_query_engine(similarity_top_k=3),
    metadata=ToolMetadata(
        name="c-elegans-research",
        description=(
            "Given a query, find the most relevant interventions for increasing the max lifespan of C. Elegans."
        ),
    ),
)

'''
Output tool
outputs list of triples: (List of 1-3 combined interventions, Explanation, Probability for what % it increases max lifespan of C.Elegans)
'''
def output_tool(interventions: str, explanation: str, max_lifespan_increase: str) -> int:
    return "Interventions: " + interventions + "\nExplanation: " + explanation + "\nMax Lifespan Increase Prediction: " + str(max_lifespan_increase)

description = """
Output a tuple of intervations, with the explanation of why it is chosen, and the probability of how much it increases the max lifespan of C. Elegans.
"""

class InterventionsOutput(BaseModel):
    interventions: str = Field(..., description="1-3 combined interventions from interventions.txt")
    explanation: str = Field(..., description="Explanation for the choice")
    max_lifespan_increase: float = Field(..., description="Multiplier prediction on how much it would increase the max lifespan of C.Elegans")


output_interventions_tool = FunctionTool.from_defaults(
    fn=output_tool,
    name="output_interventions_tool",
    description=description,
    fn_schema=InterventionsOutput,
)

instructions = """
You are helping longevity researchers choose promising life extending interventions for C. Elegans.
The proposed interventions should be a combination of 1-3 interventions that are listed in the interventions.txt file that you can read with the code interpreter.

You have acccess to a database of research papers on C. Elegans via the c_elegans_tool.

Read all the longevity interventions research papers. 
Interpolate from the experiments, hypotheses and results of the paper to propose novel interventions to prolong the lifespan of C. Elegans.

Then, reference check the interventions you propose with the uploaded csv files by writing code to check if they have been proposed before.
Update your hypotheses based on the results of the reference check. Do additional literature review if necessary with the c_elegans_tool.

Based on the data, propose the most promising interventions to prolong the lifespan of C. Elegans.
Each suggestion should include a rationale for its potential efficacy and estimated probabilities of lifespan extension in C.Elegans.
The Assistant ensures that all recommendations are evidence-based and reflect the latest research insights.

You should use the output_interventions_tool to output your proposed interventions in a structured format. Return the structured format at the end.
"""

agent = OpenAIAssistantAgent.from_new(
    name="Longevity Scientist Assistant (llama index) - 9",
    instructions=instructions,
    tools=[c_elegans_tool, output_interventions_tool],
    verbose=True,
    run_retrieve_sleep_time=1.0,
    openai_tools=[{"type": "code_interpreter"}],
    files=["./c-elegans-data/interventions.txt", "./c-elegans-data/DrugAge-database.csv"],
)

def create_agent():
    return agent