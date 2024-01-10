import logging
from typing import Any, Dict, Iterator, List, Optional

import uvicorn
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.catalog import CatalogInfo
from databricks.sdk.service.compute import ClusterInfo
from databricks.sdk.service.ml import Model
from databricks.sdk.service.sql import Dashboard
from fastapi import FastAPI
from langchain.llms import Databricks
from pydantic import BaseModel

from app.config import config
from app.models import Answer


class DatabricksManager(BaseModel):
    """
    Databricks Manager helps you to manage Databricks resources.
    """

    llm: Optional[Databricks] = None
    workspace: WorkspaceClient = None
    clean_job: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_or_create_databricks_workspace(self) -> WorkspaceClient:
        """
        Get or create a Databricks workspace.
        :return: the configured Databricks workspace
        """
        if not self.workspace:
            self.workspace = WorkspaceClient(
                host=config.DATABRICKS_SERVER_HOSTNAME, token=config.DATABRICKS_TOKEN
            )
        return self.workspace

    def submit_notebook_job_to_cluster(self, question: str) -> Answer:
        """
        Submit a Delta-Buddy job to Databricks.

        :param question: the question to ask.
        :return: the answer to the question.
        """
        name = "delta-buddy-job-from-sdk"
        logging.info(f"Submitting a job to Databricks with name: {name}")
        notebook_run = (
            self._get_or_create_databricks_workspace()
            .jobs.submit(
                run_name=name,
                tasks=[
                    jobs.JobTaskSettings(
                        description="Delta Buddy job",
                        existing_cluster_id=config.DATABRICKS_CLUSTER_ID,
                        notebook_task=jobs.NotebookTask(
                            notebook_path=config.DATABRICKS_NOTEBOOK_PATH,
                            base_parameters={
                                "question": question,
                                "serving_mode": config.DATABRICKS_SERVING_MODE.NOTEBOOK_API.value,
                            },
                        ),
                        task_key="Delta_Buddy_Task",
                        timeout_seconds=0,
                    )
                ],
            )
            .result()
        )
        response = self.workspace.jobs.get_run_output(
            run_id=notebook_run.tasks[0].run_id
        )
        if self.clean_job:
            self.workspace.jobs.delete(job_id=notebook_run.job_id)
        return Answer(
            answer=response.notebook_output.result.capitalize(),
            question=question,
        )

    @classmethod
    async def launch_llm_fast_api_from_notebook(
        cls,
        app: FastAPI,
        host: str = "0.0.0.0",
        port: int = config.DATABRICKS_LLM_PORT,
        uvicorn_config: Dict[Any, Any] = {},
    ):
        """
        Launch a Fast API from a notebook using llm with langchain.

        :param host: the host to use
        :param uvicorn_config: the extended configuration of uvicorn
        :return:
        """
        logging.info(
            f"Starting Fast API server with host[{host}], port[{port}], conf[{uvicorn_config}]"
        )
        config = uvicorn.Config(app, host=host, port=port, **uvicorn_config)
        server = uvicorn.Server(config)
        await server.serve()

    def list_alerts(self):
        """
        List alerts of Databricks account.

        :return: the alerts metadata
        """
        return self._get_or_create_databricks_workspace().alerts.list()

    def list_clusters(self) -> List[ClusterInfo]:
        """
        List clusters of Databricks account.

        :return: the clusters metadata
        """
        return self._get_or_create_databricks_workspace().clusters.list()

    def list_catalog(self, show_errors: bool = False) -> List[CatalogInfo]:
        """
        List catalog of Databricks account.

        :param show_errors: show the errors if any.
        :return: the catalog metadata
        """
        try:
            return self._get_or_create_databricks_workspace().catalogs.list()
        except Exception as e:
            if show_errors:
                logging.info(e)
            pass
            return list()

    def list_models(self) -> Iterator[Model]:
        """
        List models of Databricks account.

        :return: the models metadata
        """
        yield from self._get_or_create_databricks_workspace().model_registry.list_models()

    def list_dashboards(self) -> Iterator[Dashboard]:
        """
        List dashboards of Databricks account.

        :return: the Databricks dashboards metadata
        """
        yield from self._get_or_create_databricks_workspace().dashboards.list()
