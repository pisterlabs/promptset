import warnings
import json
import re
import pandas as pd
import requests
import ast
import numpy as np
from urllib3.exceptions import InsecureRequestWarning
from numba.core.errors import NumbaDeprecationWarning
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import demisto_client
import demisto_client.demisto_api
from demisto_client.demisto_api.rest import ApiException

from app.helper import Decorator

warnings.filterwarnings('ignore', category=InsecureRequestWarning)
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)


class LocalTextEmbedding:
    """
    A class to represent a collection of text embeddings.

    ...

    Attributes
    ----------
    file_path : str
        a string path for the csv file containing the text embeddings.
    embedding_model : str
        a string representing the name of the embedding model used.
    df : pandas.DataFrame
        a pandas DataFrame containing the loaded text embeddings.

    Methods
    -------
    get_embedding_vectors(question)
        Generates an embedding vector for the given question.
    get_similarities(question_vector)
        Calculates and adds a new 'similarities' column to the DataFrame with the cosine similarities of the question.
    get_top_similar_rows(num_rows=2)
        Returns the top similar rows from the DataFrame sorted by the 'similarities' column.
    """
    def __init__(self, text_embedding_path, embedding_model="text-embedding-ada-002"):
        """
        Constructs all the necessary attributes for the textEmbedding object.

        Parameters
        ----------
            text_embedding_path : str
                path of the csv file containing the text embeddings.
            embedding_model : str
                name of the embedding model to use.
        """

        self.file_path = text_embedding_path
        self.embedding_model = embedding_model
        self.df = pd.read_csv(text_embedding_path, usecols=['embedding', 'name'])

    def get_embedding_vectors(self, question):
        """
        Generates an embedding vector for the given question using the specified embedding model.

        Parameters
        ----------
        question : str
            The question to generate the embedding vector for.

        Returns
        -------
        numpy.ndarray
            The embedding vector for the question.
        """
        question_vector = get_embedding(question, engine=self.embedding_model)
        return question_vector

    def get_similarities(self, question_vector, num_rows=2):
        """
        Calculates and adds a new 'similarities' column to the DataFrame with the cosine similarities of each text
        in the DataFrame to the given question vector.

        Parameters
        ----------
        question_vector : numpy.ndarray
            The embedding vector of the question to compare with.
        """
        self.df["similarities"] = self.df['embedding'].apply(lambda x: cosine_similarity(np.array(ast.literal_eval(x)),
                                                                                         question_vector))
        similar_rows = self.df.sort_values(by='similarities', ascending=False).head(num_rows)
        return similar_rows


class PineConeTextEmbedding:
    """
    A class to represent a collection of text embeddings.

    ...

    Attributes
    ----------
    file_path : str
        a string path for the csv file containing the text embeddings.
    embedding_model : str
        a string representing the name of the embedding model used.
    df : pandas.DataFrame
        a pandas DataFrame containing the loaded text embeddings.

    Methods
    -------
    get_embedding_vectors(question)
        Generates an embedding vector for the given question.
    get_similarities(question_vector)
        Calculates and adds a new 'similarities' column to the DataFrame with the cosine similarities of the question.
    get_top_similar_rows(num_rows=2)
        Returns the top similar rows from the DataFrame sorted by the 'similarities' column.
    """
    def __init__(self, embedding_index, embedding_model="text-embedding-ada-002"):
        """
        Constructs all the necessary attributes for the textEmbedding object.

        Parameters
        ----------
            text_embedding_path : str
                path of the csv file containing the text embeddings.
            embedding_model : str
                name of the embedding model to use.
        """
        self.embedding_index = embedding_index
        self.embedding_model = embedding_model

    def get_embedding_vectors(self, question):
        """
        Generates an embedding vector for the given question using the specified embedding model.

        Parameters
        ----------
        question : str
            The question to generate the embedding vector for.

        Returns
        -------
        numpy.ndarray
            The embedding vector for the question.
        """
        question_vector = get_embedding(question, engine=self.embedding_model)
        return question_vector

    def get_similarities(self, question_vector, top_k=2):
        """
        Calculates and adds a new 'similarities' column to the DataFrame with the cosine similarities of each text
        in the DataFrame to the given question vector.

        Parameters
        ----------
        question_vector : numpy.ndarray
            The embedding vector of the question to compare with.
        """
        similar_rows = self.embedding_index.query(vector=question_vector, top_k=top_k)
        return similar_rows


class SOARClient:
    ERROR_ENTRY_TYPE = 4
    DEBUG_FILE_ENTRY_TYPE = 16
    SECTIONS_HEADER_REGEX = re.compile(
        r"^(Context Outputs|Human Readable section|Raw Response section)"
    )
    RAW_RESPONSE_HEADER = re.compile(r"^Raw Response section")
    CONTEXT_HEADER = re.compile(r"Context Outputs:")
    HUMAN_READABLE_HEADER = re.compile(r"Human Readable section")
    FULL_LOG_REGEX = re.compile(r".*Full Integration Log")

    def __init__(self, url, api_key, verify_ssl):
        self.url = url
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            'Authorization': api_key
        }

    def _get_playground_id(self):
        """Retrieves Playground ID from the remote XSOAR instance."""
        def playground_filter(page: int = 0):
            return {"filter": {"type": [9], "page": page}}

        client = demisto_client.configure(base_url=self.url, api_key=self.api_key, verify_ssl=self.verify_ssl)
        answer = client.search_investigations(filter=playground_filter())
        if answer.total == 0:
            raise RuntimeError("No playgrounds were detected in the environment.")
        elif answer.total == 1:
            result = answer.data[0].id
        else:
            # if found more than one playground, try to filter to results against the current user
            user_data, response, _ = client.generic_request(
                path="/user",
                method="GET",
                content_type="application/json",
                response_type=object,
            )
            if response != 200:
                raise RuntimeError("Cannot find username")
            username = user_data.get("username")

            def filter_by_creating_user_id(playground):
                return playground.creating_user_id == username
            playgrounds = list(filter(filter_by_creating_user_id, answer.data))
            for i in range(int((answer.total - 1) / len(answer.data))):
                playgrounds.extend(
                    filter(
                        filter_by_creating_user_id,
                        client.search_investigations(
                            filter=playground_filter(i + 1)
                        ).data,
                    )
                )
            if len(playgrounds) != 1:
                raise RuntimeError(
                    f"There is more than one playground to the user. "
                    f"Number of playgrounds is: {len(playgrounds)}"
                )
            result = playgrounds[0].id
        return result

    def _run_query(self, playground_id: str, query):
        """Runs a query on XSOAR instance and prints the output.

        Args:
            playground_id: The investigation ID of the playground.

        Returns:
            list. A list of the log IDs if debug mode is on, otherwise an empty list.
        """
        update_entry = {"investigationId": playground_id, "data": query}
        client = demisto_client.configure(base_url=self.url, api_key=self.api_key, verify_ssl=self.verify_ssl)
        answer = client.investigation_add_entries_sync(update_entry=update_entry)

        if not answer:
            print ("User command did not run, make sure it was written correctly.")
        log_ids = []

        for entry in answer:
            # answer should have entries with `contents` - the readable output of the command
            if entry.parent_content:
                print("[yellow]### Command:[/yellow]")
            if entry.contents:
                print("[yellow]## Readable Output[/yellow]")
                if entry.type == self.ERROR_ENTRY_TYPE:
                    print(f"[red]{entry.contents}[/red]\n")
                else:
                    print(f"{entry.contents}\n")

            # and entries with `file_id`s defined, that is the fileID of the debug log file
            if entry.type == self.DEBUG_FILE_ENTRY_TYPE:
                log_ids.append(entry.id)
        return log_ids

    @property
    def up(self):
        try:
            requests.get(self.url + "/health", headers=self.headers, verify=self.verify_ssl)
        except Exception as e:
            print("Error Occurred. " + str(e.args))
            return False
        else:
            return True

    def create_incident(self, incident_type, incident_owner, incident_name, incident_severity, incident_detail):
        data = {
            "type": incident_type,
            "name": incident_name,
            "details": incident_detail,
            "severity": incident_severity,
            "owner": incident_owner,
            "createInvestigation": True
            }
        try:
            response_api = requests.post(self.url + "/incident", headers=self.headers, data=json.dumps(data),
                                         verify=self.verify_ssl)
        except Exception as e:
            print("Error Occurred. " + str(e.args))
            return str(e.args)
        else:
            return response_api.text

    def search_incident(self, data):
        try:
            response_api = requests.post(self.url + "/incidents/search", headers=self.headers,
                                         data=json.dumps(data), verify=self.verify_ssl)
        except Exception as e:
            print("Error Occurred. " + str(e.args))
            return str(e.args)
        else:
            if response_api.status_code == 200:
                return response_api.text
            else:
                return response_api.status_code

    def execute_command(self, command: str, output_path: list, return_type: str = 'wr'):
        """
        This function executes a specific command on the Demisto client and retrieves the result. It also allows to
        specify the output path from where to fetch the execution results and return types.

        Parameters
        ----------
        command : str
            The command to be executed on the Demisto client.

        output_path : list
            A list of output paths from where to fetch the command execution results. The function also deletes
            the context entry of each output path before the execution.

        return_type : str, optional
            The return type of the command execution result. It can be 'both' (default), 'context' or 'wr'.

            'both' - Both the output context and war room entries.
            'context' - Output context only.
            'wr' - War room entries only.

        Returns
        -------
        If return_type is 'both', it returns a tuple of (output_context, war_room_entries).
        If return_type is 'context', it returns a list of output_context.
        If return_type is 'wr', it returns a list of war_room_entries.

        Example
        -------
        >>> execute_command('!ip ip="8.8.8.8"', ["AutoFocus", "IPinfo"])
        [{'IndicatorType': 'IP', 'IndicatorValue': '8.8.8.8', 'ASN': 'AS15169', 'Country': 'US', ...}]
        """
        client = demisto_client.configure(base_url=self.url, api_key=self.api_key, verify_ssl=self.verify_ssl)
        playground_id = self._get_playground_id()
        wr_entries = []
        output_context = []

        if output_path == ['-'] or "WarRoomOutput" in output_path:
            output_path = []
        try:
            if output_path:
                for output in output_path:
                    delete_context_entry = demisto_client.demisto_api.UpdateEntry(data=f"!DeleteContext key={output}",
                                                                                  investigation_id=playground_id)
                    client.investigation_add_entries_sync(update_entry=delete_context_entry)
            update_entry = demisto_client.demisto_api.UpdateEntry(data=command,
                                                                  investigation_id=playground_id)
            wr_entries = client.investigation_add_entries_sync(update_entry=update_entry)
        except ApiException as e:
            print("Exception when calling DefaultApi->investigation_add_entries_sync: %s\n" % e)
        for output in output_path:
            context_query = {"query": "${"+output+"}"}
            context = client.generic_request(
                f"investigation/{playground_id}/context", "POST", context_query
            )[0]
            output_context.append(context)

        if return_type == 'both':
            return output_context, wr_entries
        elif return_type == 'context':
            return output_context
        else:
            return wr_entries

    def enrich_indicator(self, indicator: dict, return_type: str):
        """
        This function is used to enrich the input indicator (i.e., domain, IP, URL, File, CVE) using
        a specific command that retrieves additional information about the indicator from a predefined
        data source like AutoFocus or IPinfo.

        Parameters
        ----------
        indicator : dict
            A dictionary that represents the indicator. The key should be one of the following types:
            'Domain', 'IP', 'URL', 'File', 'CVE'. The value is a list of indicators to be enriched.

        return_type : str
            The return type of the command execution result. It can be 'entry' (default), 'contents' or 'both'.

            'entry' - Entry context only (default).
            'contents' - Entry contents (raw response) only.
            'both' - Both entry context and entry contents.

        Returns
        -------
        results : list
            A list of enriched indicators represented as dictionaries. Each enriched indicator includes
            the original data from the input indicator plus the additional information retrieved by the
            executed command. If the indicator type doesn't match any predefined type ('Domain', 'IP',
            'URL', 'File', 'CVE'), no enrichment will be made and the function will return an empty list.

        Example
        -------
        >>> enrich_indicator({'IP': ['8.8.8.8']}, 'entry')
        [{'IndicatorType': 'IP', 'IndicatorValue': '8.8.8.8', 'ASN': 'AS15169', 'Country': 'US', ...}]
        """
        results = []
        if indicator.get('Domain'):
            result = {}
            for domain in indicator.get('Domain'):
                enriched_entity = self.execute_command(command=f'!domain domain="{domain}"', output_path=["AutoFocus"],
                                                       return_type=return_type)
                for dictionary in enriched_entity:
                    result.update(ast.literal_eval(dictionary)['Domain'])
                results.append(Decorator.clean_dict(result))
        if indicator.get('IP'):
            result = {}
            for ip in indicator.get('IP'):
                enriched_entity = self.execute_command(command=f'!ip ip="{ip}"', output_path=["AutoFocus", "IPinfo"],
                                                       return_type=return_type)
                for dictionary in enriched_entity:
                    result.update(ast.literal_eval(dictionary)['IP'])
                results.append(Decorator.clean_dict(result))
        if indicator.get('URL'):
            result = {}
            for url in indicator.get('URL'):
                enriched_entity = self.execute_command(command=f'!url url="{url}"', output_path=["AutoFocus"],
                                                       return_type=return_type)
                for dictionary in enriched_entity:
                    result.update(ast.literal_eval(dictionary)['URL'])
                results.append(Decorator.clean_dict(result))
        if indicator.get('File'):
            for file in indicator.get('File'):
                result = {}
                enriched_entity = self.execute_command(command=f'!file file="{file}"', output_path=["File"],
                                                       return_type=return_type)
                for dictionary in enriched_entity:
                    result.update(ast.literal_eval(dictionary)['File'])
                results.append(Decorator.clean_dict(result))
        if indicator.get('CVE'):
            for cve in indicator.get('CVE'):
                result = {}
                enriched_entity = self.execute_command(command=f'!cve cve_id="{cve}"', output_path=["CVE"],
                                                       return_type=return_type)
                for dictionary in enriched_entity:
                    result.update(ast.literal_eval(dictionary)['CVE'])
                results.append(Decorator.clean_dict(result))
        return results

