# # WARNING: Do not use this file. It is not currently in use and is not up to date.


# import logging
# from typing import Union

# from postcode.ai_services.summarizer.openai_summarizer import OpenAIReturnContext
# from postcode.ai_services.summarizer.summarization_context import Summarizer

# from postcode.types.postcode import ModelType

# from postcode.models.models import (
#     ClassModel,
#     DependencyModel,
#     FunctionModel,
#     ImportModel,
#     ModuleModel,
#     StandaloneCodeBlockModel,
# )

# # ModelType = Union[
# #     ModuleModel,
# #     ClassModel,
# #     FunctionModel,
# #     StandaloneCodeBlockModel,
# # ]


# class StandardSummarizationManager:
#     """
#     DO NOT USE: Needs to be updated based on new model structure.

#     Manages the summarization process for Python code modules.

#     This manager handles the summarization of code blocks within Python module models. It uses an instance of a
#     Summarizer to generate summaries for each code block, tracking token usage for cost estimation and updating
#     module models with their respective summaries.

#     Args:
#         - module_models_tuple (tuple[ModuleModel, ...]): A tuple of module models to summarize.
#         - summarizer (Summarizer): An instance of a Summarizer to perform the summarization.

#     Attributes:
#         - module_models_tuple (tuple[ModuleModel, ...]): Stores the module models to be summarized.
#         - summarizer (Summarizer): The summarizer instance used for generating summaries.
#         - summarized_code_block_ids (set[str]): A set of IDs for code blocks that have been summarized to avoid
#             repetition.
#         - prompt_tokens (int): The total number of prompt tokens used in summarization.
#         - completion_tokens (int): The total number of completion tokens used in summarization.
#         - updated_module_models (list[ModuleModel]): A list of module models with updated summaries.

#     Methods:
#         - `create_and_add_summaries_to_models`: Generates summaries for each module model.

#     Examples:
#         ```Python
#         client = OpenAI()
#         # Create a summarizer instance with the OpenAI client
#         summarizer = OpenAISummarizer(client=client)
#         # Create a summarization manager instance with the summarizer
#         summarization_manager = SummarizationManager(module_models_tuple, summarizer)
#         # Generate summaries for each module model
#         updated_module_models = summarization_manager.create_and_add_summaries_to_models()
#         ```
#     """

#     def __init__(
#         self,
#         models_tuple: tuple[ModuleModel, ...],
#         summarizer: Summarizer,
#     ) -> None:
#         self.models_tuple: tuple[ModuleModel, ...] = models_tuple
#         self.summarizer: Summarizer = summarizer
#         self.summarized_code_block_ids: set[str] = set()
#         self.prompt_tokens: int = 0
#         self.completion_tokens: int = 0
#         self.updated_module_models: list[ModuleModel] = []

#     @property
#     def total_cost(self) -> float:
#         """Provides the total cost of the summarization process."""
#         prompt_cost: int = self.prompt_tokens * 1  # Costs 1 cent per 1,000 tokens
#         completion_cost: int = (
#             self.completion_tokens * 3
#         )  # Costs 3 cents per 1,000 tokens
#         return (prompt_cost + completion_cost) / 100_000  # Convert to dollars

#     def create_summarizes_and_return_updated_models(self) -> tuple[ModuleModel, ...]:
#         """
#         DO NOT USE: Needs to be updated based on new model structure.

#         Generates summaries for each module model and updates them.

#         This method iterates over the provided module models, generating summaries for each. The summarized modules are then added to the list of updated module models.

#         Returns:
#             - tuple[ModuleModel, ...]: A tuple of module models with updated summaries.

#         Example:
#             ```Python
#             summarization_manager = SummarizationManager(...)
#             updated_modules = summarization_manager.create_and_add_summaries_to_models()
#             print(updated_modules)
#             ```
#         """
#         for model in self.models_tuple:
#             self._summarize_module(model)

#         return tuple(self.updated_module_models)

#     def _summarize_module(self, module_model: ModuleModel) -> None:
#         """
#         Summarizes a single module model by calling `_summarize_code_block` method and adds it to the list of
#         updated module models.
#         """
#         if module_model.id not in self.summarized_code_block_ids:
#             self._summarize_code_block(module_model)
#             # logging.info(f"Summarized module: {module_model.id}")
#             self.summarized_code_block_ids.add(module_model.id)
#             self.updated_module_models.append(module_model)

#     def _summarize_code_block(
#         self,
#         model: ModelType,
#         recursion_path: list[str] = [],
#     ) -> str | None:
#         """
#         Recursively summarizes a code block, its children, and its dependencies.

#         Travels down the tree of code blocks until it finds ones that have no children or dependencies summarizes that code block
#         and returns the summary. The return summary is then added to the summary of the parent code block to allow for better contextual
#         information in the summary. If a code block has already been summarized, the summary will be gotten from the code block model
#         and added to the prompt for generating the parent summary.

#         Args:
#             - model (ModelType): The code block model to summarize.
#             - recursion_path (list[str]): A list of code block IDs that have been visited to avoid infinite recursion.

#         Returns:
#             - str | None: The summary of the provided code block, or None if the summarization failed.

#         Notes:
#             - This method is too large and needs to be refactored.
#             - We plan to allow it to take a `recursion_path` argument to allow for customization of summary creation direction.
#         """
#         if model.id in recursion_path or not model.code_content:
#             return None
#         if model.id in self.summarized_code_block_ids:
#             return model.summary

#         recursion_path.append(model.id)

#         child_summary_list: list[str] | None = None
#         if model.children_ids:
#             child_summary_list = self._get_child_summaries(model, recursion_path)

#         dependency_summary_list: list[str] = []
#         import_details: str | None = None
#         if model.dependencies:
#             for dependency in model.dependencies:
#                 if isinstance(dependency, DependencyModel) and dependency.code_block_id:
#                     if module_local_dependency_summary := self._get_local_dependency_summary(
#                         dependency, model, recursion_path
#                     ):
#                         dependency_summary_list.append(module_local_dependency_summary)

#                 if isinstance(dependency, ImportModel):
#                     if dependency.import_module_type == "LOCAL":
#                         if not dependency.import_names:
#                             if module_import_dependency := self._get_local_import_summary(
#                                 dependency, recursion_path
#                             ):
#                                 dependency_summary_list.append(module_import_dependency)
#                         else:
#                             if import_from_dependency := self._get_local_import_from_summary(
#                                 dependency, recursion_path
#                             ):
#                                 dependency_summary_list.append(import_from_dependency)
#                     else:
#                         import_detail: str | None = self._get_import_details(dependency)
#                         if not import_detail:
#                             continue
#                         if not import_details:
#                             import_details = ""
#                         import_details += f"\n{import_detail}"

#         if isinstance(model, ModuleModel) and recursion_path:
#             dependency_summary_list, import_details = self._handle_module_model(
#                 model, recursion_path
#             )

#         children_summaries: str | None = self._stringify_children_summaries(
#             child_summary_list
#         )
#         dependency_summaries: str | None = self._stringify_dependencies_summaries(
#             dependency_summary_list
#         )

#         summary_context: OpenAIReturnContext | None = (
#             self.summarizer.test_summarize_code(
#                 model.code_content,
#                 model_id=model.id,
#                 children_summaries=children_summaries,
#                 dependency_summaries=dependency_summaries,
#                 import_details=import_details,
#             )
#         )

#         if isinstance(summary_context, OpenAIReturnContext):
#             if summary_context.summary:
#                 model.summary = summary_context.summary
#                 self.summarized_code_block_ids.add(model.id)
#                 recursion_path.remove(model.id)

#                 self.prompt_tokens += summary_context.prompt_tokens
#                 self.completion_tokens += summary_context.completion_tokens
#                 logging.info(f"Summarized code block: {model.id}")
#                 logging.info(f"Total cost: {self.total_cost}")

#         return (
#             summary_context.summary
#             if isinstance(summary_context, OpenAIReturnContext)
#             else summary_context
#         )

#     def _handle_module_model(
#         self, model: ModuleModel, recursion_path: list[str]
#     ) -> tuple[list[str], str | None]:
#         """Handles the special case of summarizing a module model."""
#         dependency_summary_list: list[str] = []
#         all_import_details: str | None = None
#         if model.imports:
#             for import_model in model.imports:
#                 if import_model.import_module_type == "LOCAL":
#                     if not import_model.import_names:
#                         if module_import := self._get_local_import_summary(
#                             import_model, recursion_path
#                         ):
#                             dependency_summary_list.append(module_import)
#                     else:
#                         if import_from := self._get_local_import_from_summary(
#                             import_model, recursion_path
#                         ):
#                             dependency_summary_list.append(import_from)
#                 else:
#                     if import_details := self._get_import_details(import_model):
#                         if not all_import_details:
#                             all_import_details = ""
#                         all_import_details += f"\n{import_details}"

#         return dependency_summary_list, all_import_details

#     def _get_import_details(self, import_model: ImportModel) -> str | None:
#         """Retrieves details of import statements to be used in the prompt."""
#         if import_model.import_module_type == "LOCAL" or not import_model.import_names:
#             return None

#         import_names_list: list[str] = []
#         for import_name in import_model.import_names:
#             if import_name.as_name:
#                 import_names_list.append(f"{import_name.name} as {import_name.as_name}")
#             else:
#                 import_names_list.append(f"{import_name.name}")

#         if import_model.imported_from:
#             import_details: str = f"from {import_model.imported_from} import {', '.join(import_names_list)}"
#         else:
#             import_details = f"import {', '.join(import_names_list)}"

#         return import_details

#     def _get_child_summaries(
#         self, model: ModelType, recursion_path: list[str]
#     ) -> list[str]:
#         """Gathers summaries of child models."""
#         child_summary_list: list[str] = []
#         if model.children_ids:
#             for child in model.children_ids:
#                 child_summary: str | None = self._summarize_code_block(
#                     child,
#                     recursion_path,
#                 )
#                 if child.summary:
#                     child_summary = child.summary
#                 else:
#                     child_summary = (
#                         f"Child ({child.id}) code content:\n{child.code_content}\n"
#                     )
#                 child_summary_list.append(child_summary)
#         return child_summary_list

#     def _stringify_children_summaries(
#         self, children_summary_list: list[str] | None
#     ) -> str | None:
#         """Converts all of the child summaries to a single string to be used in the prompt."""
#         if not children_summary_list:
#             return None

#         children_summaries: str = ""
#         for child_summary in children_summary_list:
#             children_summaries += f"\n{child_summary}"
#         return children_summaries

#     def _stringify_dependencies_summaries(
#         self, dependencies_summary_list: list[str] | None
#     ) -> str | None:
#         """Converts all of the dependency summaries to a single string to be used in the prompt."""
#         if not dependencies_summary_list:
#             return None

#         dependency_summaries: str = ""
#         for dependency_summary in dependencies_summary_list:
#             dependency_summaries += f"\n{dependency_summary}"
#         return dependency_summaries

#     def _get_local_dependency_summary(
#         self,
#         dependency: DependencyModel,
#         model: ModelType,
#         recursion_path: list[str],
#     ) -> str | None:
#         """Gets a summary for a dependency local to the module."""
#         if not model.children_ids:
#             return None

#         for child_model in model.children_ids:
#             if child_model.id == dependency.code_block_id:
#                 dependency_summary = self._summarize_code_block(
#                     child_model,
#                     recursion_path,
#                 )
#                 if dependency_summary:
#                     return dependency_summary
#                 else:
#                     return f"Dependency ({child_model.id}) code content:\n{child_model.code_content}\n"

#     def _get_local_import_summary(
#         self, dependency: ImportModel, recursion_path: list[str]
#     ) -> str | None:
#         """Gets the summary of a dependency imported from a separate module, but is local to the project."""
#         for module_model in self.models_tuple:
#             if module_model.id == dependency.local_module_id:
#                 import_summary = self._summarize_code_block(
#                     module_model,
#                     recursion_path,
#                 )
#                 if import_summary:
#                     return import_summary
#                 else:
#                     return f"Module import ({module_model.id}) code content:\n{module_model.code_content}\n"

#     def _get_local_import_from_summary(
#         self, dependency: ImportModel, recursion_path: list[str]
#     ) -> str | None:
#         """
#         Gets the summary of a dependency imported from a separate module, but is local to the project.
#         Unique handling for 'from' import statements.
#         """
#         for import_name in dependency.import_names:
#             for module_model in self.models_tuple:
#                 if module_model.id == dependency.local_module_id:
#                     if module_model.children_ids:
#                         for child_model in module_model.children_ids:
#                             if (
#                                 child_model.id == import_name.local_block_id
#                                 and child_model.id
#                             ):
#                                 import_summary = self._summarize_code_block(
#                                     child_model,
#                                     recursion_path,
#                                 )
#                                 if import_summary:
#                                     return import_summary
#                                 else:
#                                     return f"Code block import ({module_model.id}) code content:\n{module_model.code_content}\n"
