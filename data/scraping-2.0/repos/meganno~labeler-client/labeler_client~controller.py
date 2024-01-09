import json
import os

from labeler_client.helpers import get_request, post_request
from labeler_client.llm_jobs import OpenAIJob
from labeler_client.prompt import PromptTemplate
from labeler_client.service import Service
from labeler_client.subset import Subset


class Controller:
    """
    The controller class manages agents and runs agent jobs.
    """

    def __init__(self, service, auth):
        """
        Init function

        Parameters
        ----------
        service : Service
            [Megagon-only] Labeler service object for the connected project.
        auth : Authentication
            [Megagon-only] Labeler authentication object.
        """
        self.__service = service
        self.__auth = auth

    def list_agents(self, created_by_filter=None):
        """
        Get the list of registered agents by their issuer IDs.

        Parameters
        ----------
        created_by_filter : list, optional
            List of user IDs to filter agents, by default None (if None, list all)

        Returns
        -------
        list
            A list of agents that are created by specified issuers.
        """
        payload = self.__service.get_base_payload()
        payload.update({"created_by": created_by_filter})
        path = self.__service.get_service_endpoint("get_agents")
        response = get_request(path, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def list_jobs(self, filter_by, filter_values, show_agent_details=False):
        """
        Get the list of jobs with querying filters.

        Parameters
        ----------
        filter_by : str
            Filter options. Must be ["agent_uuid" | "issued_by" | "uuid"] | None
        filter_values : list
            List of uuids of entity specified in 'filter_by'
        show_agent_details : bool, optional
            If True, return agent configuration, by default False

        Returns
        -------
        list
            A list of jobs that match given filtering criteria.
        """
        payload = self.__service.get_base_payload()
        payload.update(
            {
                "details": show_agent_details,
                "filter_by": filter_by,
                "filter_values": filter_values,
            }
        )
        path = self.__service.get_service_endpoint("get_jobs")
        response = get_request(path, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def list_jobs_of_agent(self, agent_uuid, show_agent_details=False):
        """
        Get the list of jobs of a given agent.

        Parameters
        ----------
        agent_uuid : str
            Agent uuid
        show_agent_details : bool, optional
            If True, return agent configuration, by default False

        Returns
        -------
        list
            A list of jobs of a given agent
        """
        payload = self.__service.get_base_payload()
        payload.update({"details": show_agent_details})
        path = self.__service.get_service_endpoint("get_jobs_of_agent").format(
            agent_uuid=agent_uuid
        )
        response = get_request(path, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def register_agent(self, model_config, prompt_template_str):
        """
        Registers an agent with backend service.

        Parameters
        ----------
        model_config : dict
            Model configuration object
        prompt_template_str : str
            Serialized prompt template

        Returns
        -------
        dict
            object with unique agent id.
        """
        payload = self.__service.get_base_payload()
        payload.update(
            {
                "model_config": model_config,
                "prompt_template": prompt_template_str,
            }
        )
        path = self.__service.get_service_endpoint("register_agent")
        response = post_request(path, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def persist_job(self, agent_uuid, job_uuid, label_name, annotation_uuid_list):
        """
        Given annoations for a subset, persit as a job for the project.

        Parameters
        ----------
        agent_uuid : str
            Agent uuid
        job_uuid : str
            Job uuid
        label_name : str
            Label name used for annotation
        annotation_uuid_list : list
            List of uuids of records that have valid annotations from the job

        Returns
        -------
        dict
            Object with job uuid and annotation count
        """
        print("\nPersisting the job :::")
        print("\nJob ID: {}".format(job_uuid))

        payload = self.__service.get_base_payload()
        payload.update(
            {
                "label_name": label_name,
                "annotation_uuid_list": annotation_uuid_list,
            }
        )
        path = self.__service.get_service_endpoint("set_job").format(
            agent_uuid=agent_uuid, job_uuid=job_uuid
        )
        response = post_request(path, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(response.text)

    def create_agent(self, model_config, prompt_template):
        """
        Validates model configs and registers a new agent.
        Returns new agent's uuid.

        Parameters
        ----------
        model_config : dict
            Model configuration object
        prompt_template : str
            PromptTemplate object

        Returns
        -------
        agent_uuid : str
            Agent uuid
        """
        # validate configs
        model_config = OpenAIJob.validate_model_config(model_config)

        # calls register_agent (with model_config, template to serializer)
        agent = self.register_agent(
            model_config, prompt_template.get_template()
        )  # service endpoint
        agent_uuid = agent["agent_uuid"]

        print("Agent registered :::")
        print("\nAgent ID: {}".format(agent_uuid))
        print("\nModel config: {}".format(model_config))
        print("\nPrompt template: ")
        print("\033[34m{}\x1b[0m".format(prompt_template.get_template()))
        return agent_uuid

    def get_agent_by_uuid(self, agent_uuid):
        """
        Returns agent model configuration, prompt template, and creator id of specified agent.

        Parameters
        ----------
        agent_uuid : str
            Agent uuid

        Returns
        -------
        dict
            A dict containing agent details.
        """
        agents = self.list_my_agents()
        for a in agents:
            if a[0] == agent_uuid:
                return {
                    "agent_uuid": a[0],
                    "model_config": json.loads(a[1]),
                    "prompt_template": a[2],
                    "created_by": a[3],
                }
        return None

    def list_my_agents(self):
        """
        Get the list of registered agents by me.

        Returns
        -------
        agents : list
            A list of agents that are created by me.
        """
        annotator_id = self.__service.get_annotator()["user_id"]
        agents = self.list_agents([annotator_id])  # service endpoint
        return agents

    def list_my_jobs(self, show_agent_details=False):
        """
        Get the list of jobs of issued by me.

        Parameters
        ----------
        show_agent_details : bool, optional
            If True, return agent configuration, by default False

        Returns
        -------
        jobs : list
            A list of jobs of issued by me.
        """
        filter_by = "issued_by"
        annotator_id = self.__service.get_annotator()["user_id"]
        filter_values = [annotator_id]
        jobs = self.list_jobs(
            filter_by, filter_values, show_agent_details
        )  # service endpoint
        return jobs

    def run_job(self, agent_uuid, subset, label_name):
        """
        Creates, runs, and persists an LLM annotation job with given agent and subset.

        Parameters
        ----------
        agent_uuid : str
            Uuid of an agent to be used for the job
        subset : Subset
            [Megagon-only] Labeler Subset object to be annotated in the job
        label_name : str
            Label name used for annotation

        Returns
        -------
        job_uuid : str
            Job uuid
        """
        # if self.project and self.agent_token:
        #     self.create_service()
        # else:
        #     raise Exception("Service cannot be created as project and token not provided")

        label_schema = self.__service.get_schemas().value(active=True)[0]["schemas"][
            "label_schema"
        ]
        records = subset.get_data_content()

        agent = self.get_agent_by_uuid(agent_uuid)
        if not agent:
            raise Exception("Agent ID: {} is invalid".format(agent_uuid))
        model_config = agent["model_config"]
        prompt_template = PromptTemplate(
            label_schema=label_schema,
            label_names=[label_name],
            template=agent["prompt_template"],
        )  # todo: read is_json_template

        print("Job issued :::")
        print("\nAgent ID: {}".format(agent_uuid))
        print("\nModel config: {}".format(model_config))
        print("\nPrompt template: ")
        print("\033[34m{}\x1b[0m".format(prompt_template.get_template()))
        # print("─" * 70)

        # assumption: api key in env; model config is openai specific
        openai_api_key = os.environ["OPENAI_API_KEY"]
        openai_organization = (
            os.environ["OPENAI_ORGANIZATION"]
            if "OPENAI_ORGANIZATION" in os.environ
            else ""
        )

        # create job class instance
        llm_job = OpenAIJob(
            label_schema, label_name, records, model_config, prompt_template
        )
        llm_job.validate_openai_api_key(openai_api_key, openai_organization)
        llm_job.preprocess()
        llm_job.get_llm_annotations()
        llm_job.post_process_annotations()

        # create job token and service
        job_auth = self.__auth.create_access_token(job=True)
        job_uuid, job_token = job_auth["uid"], job_auth["token"]
        job_service = Service(
            project=self.__service.project, host=self.__service.host, token=job_token
        )

        # set annotations and labels for job
        job_subset = Subset(job_service, subset.get_uuid_list())
        for uuid, annotation in llm_job.annotations:
            job_subset.set_annotations(uuid, annotation)
        ret = job_service.submit_annotations(
            job_subset, llm_job.uuids_with_valid_annotations
        )
        # print("--------------------")
        # print(ret)
        annotation_uuid_list = [r["annotation_uuid"] for r in ret]

        # set job
        ret = self.persist_job(agent_uuid, job_uuid, label_name, annotation_uuid_list)
        print("\n", ret)
        return job_uuid
