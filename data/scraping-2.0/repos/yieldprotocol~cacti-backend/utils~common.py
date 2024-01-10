import inspect
import os
import json
import yaml

from web3 import Web3
import tiktoken
from langchain.llms import OpenAI
from ens import ENS
import functools
import traceback
import context

from .constants import OPENAI_API_KEY, TENDERLY_FORK_URL, CHAIN_ID_TO_NETWORK_NAME, USE_CLIENT_TO_ESTIMATE_GAS
from .evaluation import get_dummy_user_info

DEFAULT_USER_INFO = {
    "Wallet Address": None,
    "ENS Domain": None,
    "Network": None,
}


def modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model."""
    
    model_token_mapping = {
        "gpt-4": 8192,
        "gpt-4-0314": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0314": 32768,
        "gpt-4-32k-0613": 32768,
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-16k": 16385,
        "gpt-3.5-turbo-16k-0613": 16385,
        "text-ada-001": 2049,
        "ada": 2049,
        "text-babbage-001": 2040,
        "babbage": 2049,
        "text-curie-001": 2049,
        "curie": 2049,
        "davinci": 2049,
        "text-davinci-003": 4097,
        "text-davinci-002": 4097,
        "code-davinci-002": 8001,
        "code-davinci-001": 8001,
        "code-cushman-002": 2048,
        "code-cushman-001": 2048,
    }

    # handling finetuned models
    if "ft-" in modelname:
        modelname = modelname.split(":")[0]

    context_size = model_token_mapping.get(modelname, None)

    if context_size is None:
        raise ValueError(
            f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(model_token_mapping.keys())
        )

    return context_size


def widgets_yaml2functions(widgets_lst):
    functions = []
    for value in widgets_lst:
        dict_ = {}
        dict_['name'] = value['_name_']
        dict_['description'] = value['description']
        dict_['parameters'] = value['parameters']
        functions.append(dict_)
    return functions


def widgets_yaml2doc(widgets_lst):
    docs = []
    for values in widgets_lst:
        doc = ""
        widget_command_name = values['_name_'].replace('_', '-')
        widget_command_params = ','.join(['{' + p + '}' for p in values['parameters']['required']])
        widget_command = f"<|{widget_command_name}({widget_command_params})|>"
        doc += f"Widget magic command: {widget_command}\n"
        doc += f"Description of widget: {values['description']}\n"
        doc += f"Required parameters:\n"
        for param_name, prop in values['parameters']['properties'].items():
            doc += "-{" + param_name + "}" + f": {prop['description']}\n"
        if len(values["return_value_description"].strip()) > 0: 
            doc += f"Return value description:\n-{values['return_value_description']}\n"
        docs.append(doc)
    return '---\n'.join(docs)


def widgets_yaml2formats(file_path):
    with open(file_path, 'r') as file:
        widgets_lst = yaml.safe_load(file)
    # reordering the params correctly
    for j in range(len(widgets_lst)): 
        widgets_lst[j]['parameters']['properties'] = dict(sorted(widgets_lst[j]['parameters']['properties'].items(),\
                                                            key=lambda pair: widgets_lst[j]['parameters']['required'].index(pair[0])))
    assert len(set([v['_name_'] for v in widgets_lst])) == len([v['_name_'] for v in widgets_lst]), "widget names aren't unique"
    return widgets_yaml2doc(widgets_lst), widgets_yaml2functions(widgets_lst)


yaml_file_path = f"{os.getcwd()}/knowledge_base/widgets.yaml"
WIDGETS, FUNCTIONS = widgets_yaml2formats(yaml_file_path)


def widget_subset(widget_names):
    filtered_widgets = []
    for widget_doc in WIDGETS.split('---\n'):
        widget_name = widget_doc.split('(')[0].split('<|')[1].replace('-', '_').strip()
        if widget_name in widget_names:
            filtered_widgets.append(widget_doc)
    return '---\n'.join(filtered_widgets)


def set_api_key():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    OpenAI.api_key = OPENAI_API_KEY


tokenizer = tiktoken.encoding_for_model("text-davinci-003")

def estimate_gas(tx):
    if USE_CLIENT_TO_ESTIMATE_GAS:
        # Return None (null) to delegate gas estimation to the client
        return None
    return hex(context.get_web3_provider().eth.estimate_gas(tx))

def get_token_len(s: str) -> int:
    return len(tokenizer.encode(s))

# Error handling
class ConnectedWalletRequired(Exception):
    pass


class FetchError(Exception):
    pass


class ExecError(Exception):
    pass


def error_wrap(fn):

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ConnectedWalletRequired:
            return "A connected wallet is required. Please connect one and try again."
        except FetchError as e:
            return str(e)
        except ExecError as e:
            return str(e)
        except Exception as e:
            traceback.print_exc()
            return f'Got exception evaluating {fn.__name__}(args={args}, kwargs={kwargs}): {e}'

    @functools.wraps(fn)
    def wrapped_generator_fn(*args, **kwargs):
        try:
            for item in fn(*args, **kwargs):
                yield item
        except ConnectedWalletRequired:
            yield "A connected wallet is required. Please connect one and try again."
        except FetchError as e:
            yield str(e)
        except ExecError as e:
            yield str(e)
        except Exception as e:
            traceback.print_exc()
            yield f'Got exception evaluating {fn.__name__}(args={args}, kwargs={kwargs}): {e}'

    if inspect.isgeneratorfunction(fn):
        return wrapped_generator_fn
    else:
        return wrapped_fn


def ensure_wallet_connected(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not context.get_wallet_address():
            raise ConnectedWalletRequired()
        return fn(*args, **kwargs)
    return wrapped_fn


def get_real_user_info(user_info: dict) -> dict:
    try:
        user_info["Wallet Address"] = context.get_wallet_address()

        web3 = context.get_web3_provider()
        ns = ENS.from_web3(web3)
        user_info["ENS Domain"] = ns.name(user_info["Wallet Address"])

        chain_id = context.get_wallet_chain_id()
        if chain_id in CHAIN_ID_TO_NETWORK_NAME: user_info["Network"] = CHAIN_ID_TO_NETWORK_NAME[chain_id]
        return user_info
    except Exception as e:
        traceback.print_exc()
        return DEFAULT_USER_INFO

DUMMY_WALLET_ADDRESS = "0x4eD15A17A9CDF3hc7D6E829428267CaD67d95F8F"
DUMMY_ENS_DOMAIN = "cacti1729.eth"
DUMMY_NETWORK = "ethereum-mainnet"

def get_user_info(eval : bool = False) -> str:
    user_info = DEFAULT_USER_INFO
    user_info_str = ""
    if eval:
        user_info = get_dummy_user_info(user_info)
    else:
        user_info = get_real_user_info(user_info)
    for k, v in user_info.items(): user_info_str += f"User {k} = {v}\n" 
    return user_info_str.strip()