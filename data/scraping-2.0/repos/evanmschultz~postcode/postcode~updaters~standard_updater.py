# DO NOT USE THIS FILE. IT IS NOT FINISHED AND WILL NOT WORK.


# from logging import Logger

# from openai import OpenAI
# from postcode.ai_services.summarizer.openai_summarizer import OpenAISummarizer
# from postcode.ai_services.summarizer.standard_summarization_manager import (
#     StandardSummarizationManager,
# )

# from postcode.databases.chroma.setup_chroma import (
#     ChromaSetupReturnContext,
#     setup_chroma,
# )
# from postcode.json_management.json_handler import JSONHandler

# from postcode.models.models import (
#     ClassModel,
#     DirectoryModel,
#     FunctionModel,
#     ModuleModel,
#     StandaloneCodeBlockModel,
# )
# from postcode.python_parser.visitor_manager.visitor_manager import (
#     VisitorManager,
#     VisitorManagerProcessFilesReturn,
# )
# from postcode.types.postcode import ModelType


# class StandardUpdater:
#     @staticmethod
#     def update_all(
#         directory: str, output_directory: str, logger: Logger
#     ) -> ChromaSetupReturnContext:
#         # ) -> None:
#         visitor_manager = VisitorManager(directory, output_directory)
#         process_files_return: VisitorManagerProcessFilesReturn = (
#             visitor_manager.process_files()
#         )

#         models_tuple: tuple[ModelType, ...] = process_files_return.models_tuple
#         client = OpenAI(max_retries=4)
#         summarizer = OpenAISummarizer(client=client)
#         summarization_manager = StandardSummarizationManager(models_tuple, summarizer)
#         finalized_models: tuple[
#             ModuleModel, ...
#         ] = summarization_manager.create_summarizes_and_return_updated_models()

#         logger.info("Summarization complete")

#         logger.info("Saving models as JSON")
#         directory_modules: dict[str, list[str]] = process_files_return.directory_modules
#         json_manager = JSONHandler(directory, directory_modules, output_directory)

#         for model in models_tuple:
#             if isinstance(model, DirectoryModel):
#                 output_path: str = model.id

#             else:
#                 output_path: str = model.file_path + model.id
#             json_manager.save_model_as_json(model, output_path)

#         json_manager.save_visited_directories()
#         logger.info("JSON save complete")

#         logger.info("Directory parsing completed.")

#         chroma_context: ChromaSetupReturnContext = setup_chroma(
#             finalized_models, logger
#         )

#         return chroma_context
