import abc
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Literal
from urllib.parse import urlparse

from dbtunnel.utils import pkill


@dataclass
class ProxySettings:
    proxy_url: str
    port: str
    url_base_path: str


def get_cloud(context: Dict[str, Any]) -> str:
    # TODO: support gcp
    if context["extraContext"]["api_url"].endswith("azuredatabricks.net"):
        return "azure"
    return "aws"


def get_cloud_proxy_settings(cloud: str, org_id: str, cluster_id: str, port: int) -> ProxySettings:
    cloud_norm = cloud.lower()
    if cloud_norm not in ["aws", "azure"]:
        raise Exception("only supported in aws or azure")
    prefix_url_settings = {
        "aws": "https://dbc-dp-",
        "azure": "https://adb-dp-",
    }
    suffix_url_settings = {
        "aws": "cloud.databricks.com",
        "azure": "azuredatabricks.net",
    }
    # org_id = self._context["tags"]["orgId"]
    org_shard = ""
    # org_shard doesnt need a suffix of "." for dnsname its handled in building the url
    # only azure right now does dns sharding
    # gcp will need this
    if cloud_norm == "azure":
        org_shard_id = int(org_id) % 20
        org_shard = f".{org_shard_id}"
    # cluster_id = self._context["tags"]["clusterId"]
    url_base_path = f"/driver-proxy/o/{org_id}/{cluster_id}/{port}/"
    return ProxySettings(
        proxy_url=f"{prefix_url_settings[cloud_norm]}{org_id}{org_shard}.{suffix_url_settings[cloud_norm]}{url_base_path}",
        port=str(port),
        url_base_path=url_base_path
    )


Flavor = Literal[
    "gradio", "fastapi", "nicegui", "streamlit", "stable-diffusion-ui", "bokeh", "flask", "dash", "solara",
    "code-server", "chainlit"]


# from langchain: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/llms/databricks.py#L86
def get_repl_context() -> Any:
    """Gets the notebook REPL context if running inside a Databricks notebook.
    Returns None otherwise.
    """
    try:
        from dbruntime.databricks_repl_context import get_context

        return get_context()
    except ImportError:
        raise ImportError(
            "Cannot access dbruntime, not running inside a Databricks notebook."
        )


def get_current_username() -> str:
    try:
        from databricks.sdk import WorkspaceClient
        return WorkspaceClient(host=get_repl_context().browserHostName,
                               token=get_repl_context().apiToken).current_user.me().user_name
    except ImportError:
        raise ImportError(
            "Please install databricks-sdk."
        )


def extract_hostname(url):
    parsed_url = urlparse(url)
    return parsed_url.hostname


class DbTunnel(abc.ABC):

    def __init__(self, port: int, flavor: Flavor):
        self._port = port
        self._flavor = flavor
        import IPython
        self._dbutils = IPython.get_ipython().user_ns["dbutils"]
        self._display_html = IPython.get_ipython().user_ns["displayHTML"]
        self._context = json.loads(self._dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
        self._org_id = self._context["tags"]["orgId"]
        self._cluster_id = self._context["tags"]["clusterId"]
        # need to do this after the context is set
        self._cloud = get_cloud(self._context)
        self._proxy_settings = get_cloud_proxy_settings(self._cloud, self._org_id, self._cluster_id,
                                                        self._port)
        self._loop = None
        self._share = False
        self._share_information = None
        self._share_trigger_callback = None

    @abc.abstractmethod
    def _imports(self):
        pass

    @abc.abstractmethod
    def _run(self):
        pass

    @abc.abstractmethod
    def _display_url(self):
        pass

    @property
    def shared(self):
        return self._share

    def inject_auth(self, host: str = None, token: str = None):
        if os.getenv("DATABRICKS_HOST") is None:
            print("Setting databricks host from context")
            os.environ["DATABRICKS_HOST"] = host or get_repl_context().browserHostName
        if os.getenv("DATABRICKS_TOKEN") is None:
            print("Setting databricks token from context")
            os.environ["DATABRICKS_TOKEN"] = token or get_repl_context().apiToken

        return self

    def inject_sql_warehouse(self, http_path: str, server_hostname: str = None, token: str = None):
        if os.getenv("DATABRICKS_SERVER_HOSTNAME") is None:
            print("Setting databricks server hostname from context")
            os.environ["DATABRICKS_SERVER_HOSTNAME"] = server_hostname or extract_hostname(
                get_repl_context().browserHostName)

        if os.getenv("DATABRICKS_TOKEN") is None:
            print("Setting databricks token from context")
            os.environ["DATABRICKS_TOKEN"] = token or get_repl_context().apiToken

        if os.getenv("DATABRICKS_HTTP_PATH") is None:
            print("Setting databricks warehouse http path")
            os.environ["DATABRICKS_HTTP_PATH"] = http_path

        return self

    def inject_env(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) != str:
                raise ValueError(f"Value for environment variable {k} must be a string")
            print(f"Setting environment variable {k}")
            os.environ[k] = v
        return self

    def run(self):
        self._imports()
        if self._share is True and self._share_trigger_callback is not None:
            import nest_asyncio
            nest_asyncio.apply()
            self._share_trigger_callback()
        if self._share is True and self._share_information is not None:
            print(f"Use this information to publicly access your app: \n{self._share_information.public_url}")
        self._run()

    # right now only ngrok is supported so auth token is required field but in future there may be devtunnels
    def share_to_internet_via_ngrok(self,
                                    *,
                                    ngrok_api_token: str,
                                    ngrok_tunnel_auth_token: str,
                                    kill_existing_processes: bool = True,
                                    kill_all_tunnel_sessions: bool = False,
                                    basic_auth: str = None,  # "databricks:password"
                                    domain: str = None,
                                    oauth_provider: str = None,
                                    oauth_allow_domains: list[str] = None):
        self._share = True
        if kill_existing_processes is True:
            try:
                pkill("ngrok")
            except KeyError:
                print("no running tunnels to kill")
        from dbtunnel.ngrok import NgrokTunnel
        ngrok_tunnel = NgrokTunnel(self._port,
                                   ngrok_tunnel_auth_token,
                                   ngrok_api_token,
                                   basic_auth=basic_auth,
                                   domain=domain,
                                   oauth_provider=oauth_provider,
                                   oauth_allow_domains=oauth_allow_domains,
                                   )

        def ngrok_callback():
            if kill_all_tunnel_sessions is True:
                ngrok_tunnel.kill_existing_sessions()
            ngrok_tunnel.run()

        self._share_trigger_callback = ngrok_callback

        return self

    def display(self):
        pass  # no op because of the annoying flickering
        # self._display_html(self._display_url())
