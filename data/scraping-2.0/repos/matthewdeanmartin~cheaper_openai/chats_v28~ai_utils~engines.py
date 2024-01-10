import openai

from chats.ai_utils.client_utils import create_client

create_client()


def list_engines():
    # list engines
    engines = openai.Engine.list()

    # print the first engine's id
    for engine in sorted(engines.data, key=lambda engine: engine["id"]):
        print(engine["id"])


if __name__ == "__main__":
    list_engines()
