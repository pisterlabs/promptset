from glob import glob
from llama_index import (
    Document,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    SummaryIndex,
    service_context,
    load_index_from_storage,
)
from glob import glob

from llama_index.query_engine.multistep_query_engine import (
    MultiStepQueryEngine,
)
from llama_index.indices.query.query_transform.base import (
            StepDecomposeQueryTransform,
        )
from llama_index import LLMPredictor

    
from llama_index.agent import FnRetrieverOpenAIAgent
from llama_index.node_parser import SimpleNodeParser

import pandas as pd
import openai
import os
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.agent import OpenAIAgent
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.query_engine import ToolRetrieverRouterQueryEngine


class MyLLMAgent:
    def __init__(self):
        self.loading_index = {}


    def load_api_key(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        

    def load_index(self, query, selected_engine):
        d = 1536
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        query_engine = {}
        for data_file in glob("Persisted/*/"):
            name = data_file.rstrip('/')
            print(name)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=name
            )
            self.loading_index[name.split('/')[-1]] = load_index_from_storage(
                storage_context=storage_context
            )

        for key, value in self.loading_index.items():
            query_engine[key] = value.as_query_engine(similarity_top_k=3)

        act_summaries_df = pd.read_csv('summaries.csv')
        act_summaries_df = act_summaries_df.replace(
            'Indian Contract Act of 1872', 'Indian Contract 1872 act'
        )
        act_summaries_df = act_summaries_df.replace(
            'Securities Contracts (Regulation) Act',
            'Securities Contracts (Regulation) Act, 1956',
        )
        act_dict = act_summaries_df.set_index('Act Name')['Summary'].to_dict()

        summary_engine_dict = {}
        for key, value in query_engine.items():
            summary_engine_dict[key] = {
                "Summary": act_dict[key],
                "QueryEngine": value
            }

        return summary_engine_dict

    def MultiDocAgentsEngine(self, user_query):
        docs = []
        df = pd.read_csv("summaries.csv")

        for i, row in df.iterrows():
            docs.append(Document(
                text=row['Summary'],
                doc_id=row['Act Name'],
            ))
        print('Documents Created', docs[0])

        parser = SimpleNodeParser()

        nodes = parser.get_nodes_from_documents(docs)
        print(nodes[0])
        summary_index = SummaryIndex(nodes, service_context=service_context)
        # Build agents dictionary
        agents = {}
        query_engines = {}

        # this is for the baseline
        acts = []
        for key in self.loading_index.items():
            acts.append(key[0])
            acts = [act.replace(',', '').replace(' ', '') for act in acts]

        for act in acts:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=f"Persisted/{act}"),
                service_context=service_context, )

            vector_query_engine = vector_index.as_query_engine()
            summary_query_engine = summary_index.as_query_engine()

            # define tools
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=vector_query_engine,
                    metadata=ToolMetadata(
                        name="vector_tool",
                        description=(
                            "Useful for questions related to specific aspects of"
                            f" {act}."
                        ),
                    ),
                ),
                QueryEngineTool(
                    query_engine=summary_query_engine,
                    metadata=ToolMetadata(
                        name="summary_tool",
                        description=(
                            "Useful for any requests that require a holistic summary"
                            f" of EVERYTHING about {act}. For questions about"
                            " more specific sections, please use the vector_tool."
                        ),
                    ),
                ),
            ]

            # build agent
            function_llm = OpenAI(model="gpt-3.5-turbo")
            agent = OpenAIAgent.from_tools(
                query_engine_tools,
                llm=function_llm,
                verbose=True,
                system_prompt=f"""\
        You are a specialized agent designed to answer queries about {act}.
        You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
        """,
            )

            agents[act] = agent
            query_engines[act] = vector_index.as_query_engine(
                similarity_top_k=4
            )

        # define tool for each document agent
        all_tools = []
        for act in acts:
            act_summary = (
                f"This content contains informations about {act}. Use"
                f" this tool if you want to answer any questions about {act}.\n"
            )
            doc_tool = QueryEngineTool(
                query_engine=agents[act],
                metadata=ToolMetadata(
                    name=f"tool_{act}",
                    description=act_summary,
                ),
            )
            all_tools.append(doc_tool)

        tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)
        obj_index = ObjectIndex.from_objects(
            all_tools,
            tool_mapping,
            VectorStoreIndex,
        )

        top_agent = FnRetrieverOpenAIAgent.from_retriever(
            obj_index.as_retriever(similarity_top_k=3),
            system_prompt=""" \
        You are an agent designed to answer legal queries about certain Acts.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

        """,
            verbose=True,
        )

        response = top_agent.query(user_query)

        return str(response)
    

    def RetrieverRouterEngine(self, summary_engine_dict, user_query):
        # Method implementation for RetrieverRouterEngine
        from llama_index import Document

        docs = []

        for key, value in summary_engine_dict.items():
            # Making document object
            doc = Document(
                text=value["Summary"],
                extra_info={'meta data': key}
            )
            print('Document Object Created')
            docs.append(doc)
            print('Document Object Appended')

        # Converting Documents to Nodes
        print('Converting Documents to Nodes')
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0), chunk_size=1024
        )
        nodes = service_context.node_parser.get_nodes_from_documents(docs)

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        print('Documents Converted to Nodes')

        # Building Summary Index
        print('Building Summary Index')
        summary_index = SummaryIndex(nodes, storage_context=storage_context)

        list_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", use_async=True
        )

        print('Summary Index Built')

        # Defining Tools
        print('Defining Tools')
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=value["QueryEngine"],
            description=f"Useful for summarization questions related to {key}",
            name="vector_tool",
        )

        print('Vector Tools Defined')

        list_tool = QueryEngineTool.from_defaults(
            query_engine=list_query_engine,
            description=(f"Useful for retrieving specific context from {key}"),
            name="list_tool",
        )

        print('List Tools Defined')

        tool_mapping = SimpleToolNodeMapping.from_objects(
            [list_tool, vector_tool]
        )
        obj_index = ObjectIndex.from_objects(
            [list_tool, vector_tool], tool_mapping, VectorStoreIndex
        )
        print('Tool Mapping Defined')

        try:
            print('Andr aya')
            query_engine = ToolRetrieverRouterQueryEngine(obj_index.as_retriever())
            print(query_engine)
            response = query_engine.query(user_query)
            print('Response Generated', response)

        except Exception as e:
            print(e)
            response = 'No Response Found'

        return str(response)

    # Other methods remain unchanged

    # def flareQueryEngine(self, user_query):

    def MultiStepEngine(self, user_query):
        # Method implementation for MultiStepEngine
        # LLM Predictor (gpt-3)
        # documents = SimpleDirectoryReader(input_files = glob("ACTS/*")).load_data()
        # print(f"Loaded {len(documents)} Pages.")

        gpt3 = OpenAI(temperature=0, model="text-davinci-003")
        service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)

        # index = VectorStoreIndex.from_documents(documents)

        # gpt-3
        step_decompose_transform_gpt3 = StepDecomposeQueryTransform(
            LLMPredictor(llm=gpt3), verbose=True
        )

        index_summary = "Used to answer questions legal questions relevant to certain Acts"
   

        #gpt-3
        query_engine = index.as_query_engine(service_context=service_context_gpt3)
        query_engine = MultiStepQueryEngine(
            query_engine=query_engine,
            query_transform=step_decompose_transform_gpt3,
            index_summary=index_summary,
        )

        response_gpt3 = query_engine.query(user_query)

        return str(response_gpt3)


# Example of usage in app.py:
# my_agent = MyLLMAgent()
# summary_dict = my_agent.load_index(query, selected_engine)
# response = my_agent.MultiDocAgentsEngine(query)
