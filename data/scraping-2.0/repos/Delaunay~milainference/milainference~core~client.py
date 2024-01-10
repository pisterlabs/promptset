import openai

from .server_lookup import select_inferences_server


def init_client(model=None):
    server = select_inferences_server(model)

    real_model = server["model"]

    openai.api_key = "EMPTY"
    openai.api_base = f"http://{server['host']}:{server['port']}/v1"

    return real_model
