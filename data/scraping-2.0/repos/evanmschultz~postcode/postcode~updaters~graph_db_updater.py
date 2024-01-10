import logging

from openai import OpenAI
from postcode.ai_services.summarizer.graph_db_summarization_manager import (
    GraphDBSummarizationManager,
)
from postcode.ai_services.summarizer.openai_summarizer import OpenAISummarizer
from postcode.ai_services.summarizer.summarization_mapper import SummarizationMapper
from postcode.databases.arangodb.arangodb_connector import ArangoDBConnector

from postcode.databases.arangodb.arangodb_manager import ArangoDBManager
from postcode.databases.chroma.chromadb_collection_manager import (
    ChromaCollectionManager,
)
import postcode.databases.chroma.chroma_setup as chroma_setup
from postcode.json_management.json_handler import JSONHandler
from postcode.models.models import (
    DirectoryModel,
    ModuleModel,
)
from postcode.python_parser.visitor_manager.visitor_manager import (
    VisitorManager,
    VisitorManagerProcessFilesReturn,
)
from postcode.types.postcode import ModelType


class GraphDBUpdater:
    """
    Graph DB based updater.

    Updates parses the files in a directory, saves the models as JSON, in the graph database, and in a ChromaDB collection.

    Args:
        - directory (str): The directory of the project to update.
        - output_directory (str): The directory to save the JSON files.
        - graph_connector (ArangoDBConnector): The ArangoDB connector to use for connecting to the graph database.
            - default: ArangoDBConnector() - instantiates a new ArangoDBConnector with its default values

    Example:
        ```Python
        from postcode.databases.arangodb.arangodb_connector import ArangoDBConnector
        from postcode.updaters.graph_db_updater import GraphDBUpdater

        # Create the ArangoDB connector.
        arango_connector = ArangoDBConnector()

        # Create the GraphDBUpdater.
        graph_updater = GraphDBUpdater(directory, output_directory, arango_connector)

        # Update all the models for the project and setup Chroma.
        chroma__collection_manager = graph_updater.update_all()
        ```
    """

    def __init__(
        self,
        directory: str,
        output_directory: str,
        graph_connector: ArangoDBConnector = ArangoDBConnector(),
    ) -> None:
        self.directory: str = directory
        self.output_directory: str = output_directory
        self.graph_connector: ArangoDBConnector = graph_connector

        self.graph_manager = ArangoDBManager(graph_connector)

    def update_all(self) -> ChromaCollectionManager:
        """
        Updates all the models for a project using the graph database.

        Note:
            This method will delete all the existing collections in the graph database, summarize every code block in the project,
            and save the new models in the graph database and as JSON. Use with caution as it is expensive with respect to time, resources,
            and money.

        Args:
            - directory (str): The directory of the project to update.
            - output_directory (str): The directory to save the JSON files.

        Returns:
            - chroma_collection_manager (ChromaDBCollectionManager): The ChromaDB collection manager.

        Raises:
            - Exception: If no finalized models are returned from summarization.

        Example:
            ```Python
            graph_updater = GraphDBUpdater(directory, output_directory)

            # Update all the models for the project and setup Chroma.
            chroma_manager = graph_updater.update_all()
            ```
        """

        self.graph_connector.delete_all_collections()
        self.graph_connector.ensure_collections()

        process_files_return: VisitorManagerProcessFilesReturn = (
            self._visit_and_parse_files(self.directory)
        )
        models_tuple: tuple[ModelType, ...] = process_files_return.models_tuple

        self._upsert_models_to_graph_db(models_tuple)

        finalized_models: list[ModelType] | None = self._map_and_summarize_models(
            models_tuple
        )

        if not finalized_models:
            raise Exception("No finalized models returned from summarization.")

        json_manager = JSONHandler(
            self.directory,
            process_files_return.directory_modules,
            self.output_directory,
        )
        self._save_json(finalized_models, json_manager)
        self._upsert_models_to_graph_db(tuple(finalized_models))

        return chroma_setup.setup_chroma_with_update(finalized_models)

    def _visit_and_parse_files(
        self, directory: str
    ) -> VisitorManagerProcessFilesReturn:
        """Visits and parses the files in the directory."""

        logging.info("Starting the directory parsing.")
        visitor_manager = VisitorManager(directory)

        return visitor_manager.process_files()

    def _get_module_ids(self, models_tuple: tuple[ModelType, ...]) -> list[str]:
        """Returns a list of module IDs from the models tuple."""

        return [model.id for model in models_tuple if isinstance(model, ModuleModel)]

    def _upsert_models_to_graph_db(self, models_tuple: tuple[ModelType, ...]) -> None:
        """Upserts the models to the graph database."""

        self.graph_manager.upsert_models(
            list(models_tuple)
        ).process_imports_and_dependencies().get_or_create_graph()

    def _save_json(self, models: list[ModelType], json_manager: JSONHandler) -> None:
        """Saves the models as JSON."""

        logging.info("Saving models as JSON")
        for model in models:
            if isinstance(model, DirectoryModel):
                output_path: str = model.id

            else:
                output_path: str = model.file_path + model.id
            json_manager.save_model_as_json(model, output_path)

        json_manager.save_visited_directories()
        logging.info("JSON save complete")

    def _map_and_summarize_models(
        self,
        models_tuple: tuple[ModelType, ...],
    ) -> list[ModelType] | None:
        """Maps and summarizes the models."""

        module_ids: list[str] = self._get_module_ids(models_tuple)
        summarization_mapper = SummarizationMapper(
            module_ids, models_tuple, self.graph_manager
        )
        client = OpenAI(max_retries=4)
        summarizer = OpenAISummarizer(client=client)
        summarization_manager = GraphDBSummarizationManager(
            models_tuple, summarization_mapper, summarizer, self.graph_manager
        )

        finalized_models: list[
            ModelType
        ] | None = summarization_manager.create_summaries_and_return_updated_models()
        logging.info("Summarization complete")

        return finalized_models if finalized_models else None
