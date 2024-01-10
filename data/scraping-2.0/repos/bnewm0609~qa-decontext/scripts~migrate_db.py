import json
import shutil
from ast import literal_eval
from pathlib import Path


from decontext.cache import DiskCache
from decontext.data_types import OpenAIChatMessage


def get_key(params: dict, is_chat_model) -> str:
    """Creates a dict that is serialized to a json string"""
    _key = {k: v for k, v in params.items() if k not in {"user", "prompt", "messages"}}
    if is_chat_model:
        _key["messages"] = [m.dict() for m in params["messages"]]
    else:
        _key["prompt"] = params["prompt"]
    return json.dumps(_key, sort_keys=True)  # sort keys for consistent serialization


def main():
    # We need to parse the arguments out of the key...
    cache = DiskCache.load()

    shutil.copyfile(Path(cache.cache_dir) / "cache.db", Path(cache.cache_dir) / "cache_backup.db")

    for key in cache._cache.keys():
        kvs = []
        is_chat_model = False
        # parse the args out of the key:
        args = key.split("-")
        for arg in args:
            # if the key doesn't have an underscore, attach it to the previous
            # value - we probably accidentally split on a hyphen.
            is_new_arg = any(
                [
                    arg.startswith(prefix)
                    for prefix in {
                        "model",
                        "temperature",
                        "stop",
                        "logprobs",
                        "prompt",
                        "top_p",
                        "max_tokens",
                        "messages",
                        "msg_",
                    }
                ]
            )
            if not is_new_arg:
                kvs[-1] = (kvs[-1][0], kvs[-1][1] + "-" + arg)
                continue

            if arg.startswith("top_p"):
                k = "top_p"
                v = arg.split("_")[-1]
            elif arg.startswith("max_tokens"):
                k = "max_tokens"
                v = arg.split("_")[-1]
            elif arg.startswith("messages"):
                k, v = arg.split("_", 1)
                is_chat_model = True
            elif arg.startswith("msg_"):
                # these are at the end, so we want to ignore everything that follows
                break
            else:
                k, v = arg.split("_", 1)

            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            if k == "stop":
                v = literal_eval(v)

            kvs.append((k, v))

        params = dict(kvs)

        if is_chat_model:
            params["messages"] = literal_eval(params["messages"])
            params["messages"] = [OpenAIChatMessage(**message) for message in params["messages"]]

        new_key = get_key(params, is_chat_model)
        print("OLD KEY\n")
        print(repr(key))
        print("\n\n=====\n\nNew Key\n")
        print(new_key)
        print("\n\n~~~~=====~~~~~\n\n")

        cache._cache[new_key] = cache._cache[key]
        del cache._cache[key]


if __name__ == "__main__":
    main()
