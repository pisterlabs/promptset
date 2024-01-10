from langchain import OpenAI
from langchain.memory import ConversationBufferMemory
from llama_index import LLMPredictor
from llama_index.indices.query.query_transform import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaToolkit, GraphToolConfig, create_llama_agent

from crystal_gpt.llama_index import LlamaIndexManager
from loguru import logger

from crystal_gpt.prompts import PromptController
from tools.articles import get_id


class Engine:
    index_manager: LlamaIndexManager
    toolkit: LlamaToolkit
    index_configs: list[IndexToolConfig]
    graph_config: GraphToolConfig

    def __init__(self):
        logger.debug("Initializing CrystalGPT Engine...")
        self.llm = OpenAI(temperature=0.5, max_tokens=384, top_p=0.8, model_name="gpt-3.5-turbo")
        self.llm_predictor = LLMPredictor(llm=self.llm)

        self.index_manager = LlamaIndexManager()

        self.query_transform = DecomposeQueryTransform(self.llm_predictor, verbose=True)
        self.query_configs = self._create_query_configs(self.query_transform)
        self.reticulate_splines()

        logger.success("Engine initialized successfully")

    def reticulate_splines(self):
        logger.debug("Reticulating splines...")
        self.graph_config = self._create_graph_config(self.index_manager.graph, self.query_configs)
        self.index_configs = self._create_index_configs(self.index_manager.documents, self.index_manager.idx_set)
        self.toolkit = LlamaToolkit(graph_configs=[self.graph_config], tool_configs=self.index_configs)
        logger.success("Toolkit and indices initialized successfully")

    async def query(self, query: str, memory: ConversationBufferMemory = None, articles_only: bool = False,
                    agent: str = None):
        logger.debug(f"Executing query: {query}")
        agent_chain = create_llama_agent(
            toolkit=self.toolkit,
            llm=self.llm,
            memory=memory or PromptController.init_conversation_memory(),
            agent=agent or ("zero-shot-react-description" if articles_only else "conversational-react-description"),
            verbose=True,
        )
        agent_answer = await agent_chain.arun(query)
        logger.info(f"Response: {agent_answer}")
        return agent_answer

    @staticmethod
    def _create_index_configs(documents, index_set):
        logger.debug("Creating index configs...")
        index_configs = []
        for y in documents:
            tool_config = IndexToolConfig(
                index=index_set[get_id(y)],
                name=f"{y['title']}",
                description=f"{y['abstract']}",
                index_query_kwargs={"similarity_top_k": 3},
                tool_kwargs={"return_direct": True},
            )
            index_configs.append(tool_config)
        logger.info(f"Created {len(index_configs)} index configs")
        return index_configs

    @staticmethod
    def _create_query_configs(decompose_transform):
        logger.debug("Creating query configs...")
        query_configs = [
            {
                "index_struct_type": "simple_dict",
                "query_mode": "default",
                "query_kwargs": {"similarity_top_k": 1, "verbose": True},
                "query_transform": decompose_transform,
            },
            {
                "index_struct_type": "list",
                "query_mode": "default",
                "query_kwargs": {
                    "response_mode": "tree_summarize",
                    "verbose": True,
                },
            },
        ]
        return query_configs

    @staticmethod
    def _create_graph_config(graph, query_configs):
        logger.debug("Creating graph config...")
        graph_config = GraphToolConfig(
            graph=graph,
            name="Intermediate Answer",
            description="Get your questions answered with authority.",
            query_configs=query_configs,
            tool_kwargs={"return_direct": False},
        )
        logger.info("Graph config created successfully")
        return graph_config
