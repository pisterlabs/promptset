import pandas as pd
from typing import List


def _get_embedding(snippet: str, embedding_model: str) -> List[int]:
    from openai.embeddings_utils import get_embedding

    return get_embedding(snippet, engine=embedding_model)


import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(
            self, url, proxies, stream, verify, cert
        )
        settings["verify"] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


def tokenize_and_embed(
    string_list: List[str],
    embedding_model: str = "text-embedding-ada-002",
    embedding_encoding: str = "cl100k_base",
    embedding_name="embedding",
    n_token_name="n_tokens",
) -> pd.DataFrame:
    import tiktoken
    from functools import partial
    from pandarallel import pandarallel

    with no_ssl_verification():
        encoding_model = tiktoken.get_encoding(embedding_encoding)

        df = pd.DataFrame({"snippet": string_list})

        df[n_token_name] = df.snippet.apply(lambda x: len(encoding_model.encode(x)))

        pandarallel.initialize(progress_bar=False)

        try:
            df[embedding_name] = df.snippet.parallel_apply(
                partial(_get_embedding, embedding_model=embedding_model)
            )
        except ValueError as e:
            print(f"failed df:\n{string_list}")
            raise

    return df
