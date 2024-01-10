import functools
import json
import re
import requests
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Generator, List, Optional, Union, Literal, TypedDict
import traceback

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.base import BaseOutputParser
from ens import ENS

import ui_workflows
import config
import context
import utils
from utils import error_wrap, ensure_wallet_connected, ConnectedWalletRequired, FetchError, ExecError, ETH_MAINNET_CHAIN_ID, get_token_balance, format_token_balance
import utils.timing as timing
from utils.coingecko.coingecko_coin_currency import coin_list, currency_list, coingecko_api_url_prefix
import registry
import streaming
from chat.container import ContainerMixin, dataclass_to_container_params
from integrations import (
    etherscan, defillama, center, opensea,
)
from ui_workflows import (
    aave, ens
)
from .index_lookup import IndexLookupTool

from ui_workflows.multistep_handler import register_ens_domain, exec_aave_operation


WIDGET_START = '<|'
WIDGET_END = '|>'

RE_COMMAND = re.compile(r"\<\|(?P<command>[^(]+)\((?P<params>[^)<{}]*)\)\|\>")


TEMPLATE = '''You are a web3 widget tool. You have access to a list of widget magic commands that you can delegate work to, by invoking them and chaining them together, to provide a response to an input query. Magic commands have the structure "<|command(parameter1, parameter2, ...)|>" specifying the command and its input parameters. They can only be used with all parameters having known and assigned values, otherwise, they have to be kept secret. The command may either have a display- or a fetch- prefix. When you return a display- command, the user will see data, an interaction box, or other inline item rendered in its place. When you return a fetch- command, data is fetched over an API and injected in place. Users cannot type or use magic commands, so do not tell them to use them. Fill in the command with parameters as inferred from the input. If there are missing parameters, do not use magic commands but mention what parameters are needed instead. If there is no appropriate widget available, explain that more information is needed. Do not make up a non-existent widget magic command, only use the applicable ones for the situation, and only if all parameters are available. You might need to use the output of widget magic commands as the input to another to get your final answer. Here are the widgets that may be relevant:
---
{task_info}
---
Use the following format:

## Widget Command: most relevant widget magic command to respond to input
## Known Parameters: input parameter-value pairs representing inputs to the above widget magic command
## Response: return the widget magic command with ALL its respective input parameter values (omit parameter names)

Tool input: {question}
## Widget Command:'''


@registry.register_class
class IndexWidgetTool(IndexLookupTool):
    """Tool for searching a widget index and figuring out how to respond to the question."""

    _chain: LLMChain
    _evaluate_widgets: bool

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        evaluate_widgets = kwargs.pop('evaluate_widgets', True)
        model_name = kwargs.pop('model_name', None)

        prompt = PromptTemplate(
            input_variables=["task_info", "question"],
            template=TEMPLATE,
        )

        new_token_handler = kwargs.get('new_token_handler')
        response_buffer = ""
        response_state = 0  # finite-state machine state
        response_prefix = "## Response:"

        def injection_handler(token):
            nonlocal new_token_handler, response_buffer, response_state, response_prefix

            timing.log('first_widget_token')

            response_buffer += token
            if response_state == 0:  # we are still waiting for response_prefix to appear
                if response_prefix not in response_buffer:
                    # keep waiting
                    return
                else:
                    # we have found the response_prefix, trim everything before that
                    timing.log('first_widget_response_token')
                    response_state = 1
                    response_buffer = response_buffer[response_buffer.index(response_prefix) + len(response_prefix):]

            if response_state == 1:  # we are going to output the response incrementally, evaluating any fetch commands
                while WIDGET_START in response_buffer and self._evaluate_widgets:
                    if WIDGET_END in response_buffer:
                        # parse fetch command
                        response_buffer = iterative_evaluate(response_buffer)
                        if isinstance(response_buffer, Callable):  # handle delegated streaming
                            def handler(token):
                                nonlocal new_token_handler
                                timing.log('first_visible_widget_response_token')
                                return new_token_handler(token)
                            response_buffer(handler)
                            response_buffer = ""
                            return
                        elif isinstance(response_buffer, Generator):  # handle stream of widgets
                            for item in response_buffer:
                                timing.log('first_visible_widget_response_token')
                                new_token_handler(str(item) + "\n")
                            response_buffer = ""
                            return
                        elif len(response_buffer.split(WIDGET_START)) == len(response_buffer.split(WIDGET_END)):
                            # matching pairs of open/close, just flush
                            # NB: for better frontend parsing of nested widgets, we need an invariant that
                            # there are no two independent widgets on the same line, otherwise we can't
                            # detect the closing tag properly when there is nesting.
                            response_buffer = response_buffer.replace(WIDGET_END, WIDGET_END + '\n')
                            break
                        else:
                            # keep waiting
                            return
                    else:
                        # keep waiting
                        return

                if 0 < len(response_buffer) < len(WIDGET_START) and WIDGET_START.startswith(response_buffer):
                    # keep waiting if we could potentially be receiving WIDGET_START
                    return
                token = response_buffer
                response_buffer = ""
                if token.strip():
                    timing.log('first_visible_widget_response_token')
                new_token_handler(token)
                if '\n' in token:
                    # we have found a line-break in the response, switch to the terminal state to mask subsequent output
                    response_state = 2

        chain = streaming.get_streaming_chain(prompt, injection_handler, model_name=model_name)
        super().__init__(
            *args,
            _chain=chain,
            _evaluate_widgets=evaluate_widgets,
            content_description="widget magic command definitions for users to invoke web3 transactions or live data when the specific user action or transaction is clear. You can look up live prices, DeFi yields, wallet balances, ENS information, token contract addresses, do transfers or swaps or deposit tokens to farm yields, or search for NFTs, and retrieve data about NFT collections, assets, trait names and trait values. It cannot help the user with understanding how to use the app or how to perform certain actions.",
            input_description="a standalone query phrase with all relevant contextual details mentioned explicitly without using pronouns in order to invoke the right widget",
            output_description="a summarized answer with relevant magic command for widget(s), or a question prompt for more information to be provided",
            **kwargs
        )

    def _run(self, query: str) -> str:
        """Query index and answer question using document chunks."""
        task_info = super()._run(query)
        example = {
            "task_info": task_info,
            "question": query,
            "stop": "User",
        }
        result = self._chain.run(example)
        return result.strip()


def iterative_evaluate(phrase: str) -> Union[str, Generator, Callable]:
    while True:
        # before we had streaming, we could use this
        #eval_phrase = RE_COMMAND.sub(replace_match, phrase)
        # now, iterate manually to find any streamable components
        eval_phrase = ""
        last_matched_char = 0
        for match in RE_COMMAND.finditer(phrase):
            span = match.span()
            eval_phrase += phrase[last_matched_char: span[0]]
            last_matched_char = span[1]
            replaced = replace_match(match)
            if isinstance(replaced, str):
                eval_phrase += replaced
            else:
                # we assume that at most one fetch widget exists if we get a stream of
                # display widgets, and short-circuit to return the stream here.
                # we also ignore any surrounding text that might be present
                return replaced
        eval_phrase += phrase[last_matched_char:]
        if eval_phrase == phrase:
            break
        phrase = eval_phrase
    return phrase


def sanitize_str(s: str) -> str:
    s = s.strip()
    if s.startswith('"') and s.endswith('"') or s.startswith("'") and s.endswith("'"):
        s = s[1:-1]

    # If parameter not detected in message, model will fill in with "None" so parse that to native None
    s = None if s.lower() == 'none' else s
    return s


def replace_match(m: re.Match) -> Union[str, Generator, Callable]:
    command = m.group('command')
    params = m.group('params')
    params = list(map(sanitize_str, params.split(','))) if params else []
    timing.log('first_widget_command')
    print('found command:', command, params)
    if command == 'fetch-nft-search':
        return fetch_nft_search(*params)
    elif command == 'fetch-price':
        return str(fetch_price(*params))
    elif command == 'fetch-nft-collection-assets-by-trait':
        return fetch_nft_search_collection_by_trait(*params, for_sale_only=False)
    elif command == 'fetch-nft-collection-assets-for-sale-by-trait':
        return fetch_nft_search_collection_by_trait(*params, for_sale_only=True)
    elif command == 'fetch-nft-collection-info':
        # return str(fetch_nft_collection(*params))
        # we also fetch some collection assets as a convenience
        return str(fetch_nft_collection_assets(*params))
    elif command == 'fetch-nft-collection-assets-for-sale':
        return fetch_nft_collection_assets_for_sale(*params)
    elif command == 'fetch-nft-collection-traits':
        return str(fetch_nft_collection_traits(*params))
    elif command == 'fetch-nft-collection-trait-values':
        return str(fetch_nft_collection_trait_values(*params))
    # elif command == 'fetch-nft-asset-info':
    #    return str(fetch_nft_asset(*params))
    elif command == 'fetch-nft-asset-traits':
        return str(fetch_nft_asset_traits(*params))
    elif command == 'fetch-nft-buy-asset':
        return str(fetch_nft_buy(*params))
    elif command == 'fetch-nfts-owned-by-address-or-domain':
        return str(fetch_nfts_owned_by_address_or_domain(*params))
    elif command == 'fetch-nfts-owned-by-user':
        return str(fetch_nfts_owned_by_user(*params))
    elif command == 'fetch-balance':
        return str(fetch_balance(*params))
    elif command == 'fetch-my-balance':
        return str(fetch_my_balance(*params))
    elif command == 'fetch-eth-in':
        return str(fetch_eth_in(*params))
    elif command == 'fetch-eth-out':
        return str(fetch_eth_out(*params))
    elif command == 'fetch-gas':
        return str(fetch_gas(*params))
    elif command == 'fetch-yields':
        return str(fetch_yields(*params))
    elif command == 'fetch-app-info':
        return fetch_app_info(*params)
    # elif command == 'fetch-scraped-sites':
    #     return fetch_scraped_sites(*params)
    elif command == 'fetch-link-suggestion':
        return fetch_link_suggestion(*params) 
    elif command == aave.AaveSupplyContractWorkflow.WORKFLOW_TYPE:
        return str(exec_aave_operation(*params, operation='supply'))
    elif command == aave.AaveBorrowContractWorkflow.WORKFLOW_TYPE:
        return str(exec_aave_operation(*params, operation='borrow'))
    elif command == aave.AaveRepayContractWorkflow.WORKFLOW_TYPE:
        return str(exec_aave_operation(*params, operation='repay'))
    elif command == aave.AaveWithdrawContractWorkflow.WORKFLOW_TYPE:
        return str(exec_aave_operation(*params, operation='withdraw'))
    elif command == 'ens-from-address':
        return str(ens_from_address(*params))
    elif command == 'address-from-ens':
        return str(address_from_ens(*params))
    elif command == ens.ENSRegistrationContractWorkflow.WORKFLOW_TYPE:
        return str(register_ens_domain(*params))
    elif command == ens.ENSSetTextWorkflow.WORKFLOW_TYPE:
        return str(set_ens_text(*params))
    elif command == ens.ENSSetPrimaryNameWorkflow.WORKFLOW_TYPE:
        return str(set_ens_primary_name(*params))
    elif command == ens.ENSSetAvatarNFTWorkflow.WORKFLOW_TYPE:
        return str(set_ens_avatar_nft(*params))
    elif command.startswith('display-'):
        return m.group(0)
    else:
        # unrecognized command, just return for now
        # assert 0, 'unrecognized command: %s' % m.group(0)
        return m.group(0)

@error_wrap
def fetch_price(basetoken: str, quotetoken: str = "usd") -> str:
    # TODO
    # Handle failures
    """
    Failures:
    - quotetoken not mentioned it can assume it to be usd or eth
    - Cannot identify duplicates in the coin list
    - Cannot handle major mispells
    """
    for c in coin_list:
        if c['id'].lower() == basetoken.lower() or \
            c['symbol'].lower() == basetoken.lower() or \
                c['name'].lower() == basetoken.lower():
            basetoken_id = c['id'].lower()
            basetoken_name = c['name']
            break
    else:
        return f"Query token {basetoken} not supported"

    if quotetoken.lower() in currency_list:
        quotetoken_id = quotetoken.lower()
    else:
        return f"Quote currency {quotetoken} not supported"

    coingecko_api_url = coingecko_api_url_prefix + f"?ids={basetoken_id}&vs_currencies={quotetoken_id}"
    response = requests.get(coingecko_api_url)
    response.raise_for_status()
    return f"The price of {basetoken_name} is {list(list(response.json().values())[0].values())[0]} {quotetoken}"


@error_wrap
def fetch_balance(token: str, wallet_address: str) -> str:
    if not wallet_address or wallet_address == 'None':
        raise FetchError(f"Please specify the wallet address to check the token balance of.")
    web3 = context.get_web3_provider()
    chain_id = context.get_wallet_chain_id()
    balance = get_token_balance(web3, chain_id, token, wallet_address)
    return format_token_balance(chain_id, token, balance)


@error_wrap
def fetch_my_balance(token: str) -> str:
    wallet_address = context.get_wallet_address()
    if not wallet_address:
        raise ConnectedWalletRequired
    return fetch_balance(token, wallet_address)


@error_wrap
def fetch_eth_in(wallet_address: str) -> str:
    return etherscan.get_all_eth_to_address(wallet_address)


@error_wrap
def fetch_eth_out(wallet_address: str) -> str:
    return etherscan.get_all_eth_from_address(wallet_address)


@error_wrap
def fetch_gas(wallet_address: str) -> str:
    return etherscan.get_all_gas_for_address(wallet_address)


@error_wrap
def fetch_app_info(query: str) -> Callable:
    def fn(token_handler):
        tool = dict(
            name="AppUsageGuideTool",
            type="tools.app_usage_guide.AppUsageGuideTool",
            _streaming=True,
        )
        tool = streaming.get_streaming_tools([tool], token_handler)[0]
        tool._run(query)
    return fn

# deprating this in favor of link suggestion as the 2 seem to clash a lot
# leaving here for future reference in case we ever want to revive
#
# @error_wrap
# def fetch_scraped_sites(query: str) -> Callable:
#     def fn(token_handler):
#         scraped_sites_index = config.initialize(config.scraped_sites_index)
#         tool = dict(
#             type="tools.index_answer.IndexAnswerTool",
#             _streaming=True,
#             name="ScrapedSitesIndexAnswer",
#             content_description="",  # not used
#             index=scraped_sites_index, 
#             top_k=3,
#             source_key="url",
#         )
#         tool = streaming.get_streaming_tools([tool], token_handler)[0]
#         tool._run(query)
#     return fn

@error_wrap
def fetch_link_suggestion(query: str) -> Callable:
    def fn(token_handler):
        dapps_index = config.initialize(config.dapps_index)
        tool = dict(
            type="tools.index_link_suggestion.IndexLinkSuggestionTool",
            _streaming=True,
            name="LinkSuggestionIndexAnswer",
            content_description="",  # not used
            index=dapps_index, 
            top_k=3,
            source_key="url",
        )
        tool = streaming.get_streaming_tools([tool], token_handler)[0]
        tool._run(query)
    return fn


class ListContainer(ContainerMixin, list):
    def message_prefix(self) -> str:
        num = len(self)
        return _get_result_list_prefix(num)

    def container_name(self) -> str:
        return 'display-list-container'

    def container_params(self) -> Dict:
        return dict(
            items=[item.struct() for item in self],
        )


def _get_result_list_prefix(num: int):
    if num > 0:
        return f"I found {num} result{'s' if num > 1 else ''}: "
    else:
        return "I did not find any results."


@dataclass
class StreamingListContainer(ContainerMixin):
    operation: str
    item: Optional[ContainerMixin] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    is_thinking: Optional[bool] = None

    def container_name(self) -> str:
        return 'display-streaming-list-container'

    def container_params(self) -> Dict:
        return dict(
            operation=self.operation,
            item=self.item.struct() if self.item else None,
            prefix=self.prefix,
            suffix=self.suffix,
            isThinking=self.is_thinking,
        )


@dataclass
class TableHeader:
    field_name: str
    display_name: str

    def container_params(self) -> Dict:
        return dataclass_to_container_params(self)


@dataclass
class TableContainer(ContainerMixin):
    headers: List[TableHeader]
    rows: List[ContainerMixin]
    message_prefix_str: str = ""

    def message_prefix(self) -> str:
        return self.message_prefix_str

    def container_name(self) -> str:
        return 'display-table-container'

    def container_params(self) -> Dict:
        return dict(
            headers=[header.container_params() for header in self.headers],
            rows=[row.struct() for row in self.rows],
        )


@error_wrap
def fetch_nft_search(search_str: str) -> Generator:
    yield StreamingListContainer(operation="create", prefix="Searching", is_thinking=True)
    num = 0
    try:
        for item in center.fetch_nft_search(search_str):
            yield StreamingListContainer(operation="append", item=item)
            num += 1
    finally:
        yield StreamingListContainer(operation="update", prefix=_get_result_list_prefix(num), is_thinking=False)


@error_wrap
def fetch_nft_search_collection_by_trait(
        network: str, address: str, trait_name: str, trait_value: str, for_sale_only: bool = False) -> Generator:
    yield StreamingListContainer(operation="create", prefix="Searching", is_thinking=True)
    num = 0
    try:
        for item in center.fetch_nft_search_collection_by_trait(
                network, address, trait_name, trait_value, for_sale_only=for_sale_only):
            yield StreamingListContainer(operation="append", item=item)
            num += 1
    finally:
        yield StreamingListContainer(operation="update", prefix=_get_result_list_prefix(num), is_thinking=False)


@error_wrap
def fetch_nft_collection(network: str, address: str) -> str:
    return str(center.fetch_nft_collection(network, address))


@error_wrap
def fetch_nft_collection_assets(network: str, address: str) -> str:
    ret = center.fetch_nft_collection_assets(network, address)
    return str(ret)


@error_wrap
def fetch_nft_collection_assets_for_sale(network: str, address: str) -> Generator:
    yield StreamingListContainer(operation="create", prefix="Searching", is_thinking=True)
    num = 0
    try:
        for item in center.fetch_nft_collection_assets_for_sale(network, address):
            yield StreamingListContainer(operation="append", item=item)
            num += 1
    finally:
        yield StreamingListContainer(operation="update", prefix=_get_result_list_prefix(num), is_thinking=False)


@error_wrap
def fetch_nft_collection_traits(network: str, address: str) -> str:
    ret = center.fetch_nft_collection_traits(network, address)
    return str(ret)


@error_wrap
def fetch_nft_collection_trait_values(network: str, address: str, trait: str) -> str:
    ret = center.fetch_nft_collection_trait_values(network, address, trait)
    return str(ret)


@error_wrap
def fetch_nft_asset(network: str, address: str, token_id: str) -> str:
    return str(center.fetch_nft_asset(network, address, token_id))


@error_wrap
def fetch_nft_asset_traits(network: str, address: str, token_id: str) -> str:
    return str(center.fetch_nft_asset_traits(network, address, token_id))


@error_wrap
def fetch_nfts_owned_by_address_or_domain(network: str, address_or_domain: str) -> str:
    return str(center.fetch_nfts_owned_by_address_or_domain(network, address_or_domain))


@error_wrap
def fetch_nfts_owned_by_user(network: str = None) -> str:
    wallet_address = context.get_wallet_address()
    parsed_network = None
    if not network:
        chain_id = context.get_wallet_chain_id()
        if chain_id == ETH_MAINNET_CHAIN_ID:
            parsed_network = "ethereum-mainnet"
        else:
            raise ExecError("Unsupported network")
    else:
        parsed_network = network
    return str(center.fetch_nfts_owned_by_address_or_domain(parsed_network, wallet_address))

@error_wrap
def fetch_nft_buy(network: str, address: str, token_id: str) -> str:
    wallet_address = context.get_wallet_address()
    nft_fulfillment_container = center.fetch_nft_buy(network, wallet_address, address, token_id)
    return str(nft_fulfillment_container)


@error_wrap
def fetch_yields(token, network, count) -> str:
    yields = defillama.fetch_yields(token, network, count)

    headers = [
        TableHeader(field_name="project", display_name="Project"),
        TableHeader(field_name="tvlUsd", display_name="TVL"),
        TableHeader(field_name="apy", display_name="APY"),
        TableHeader(field_name="apyAvg30d", display_name="30 day Avg. APY")
    ]

    if network == '*':
        headers = [TableHeader(field_name="network", display_name="Network")] + headers

    if token == '*':
        headers = [TableHeader(field_name="token", display_name="Token")] + headers

    table_container = TableContainer(headers=headers, rows=yields)
    return str(table_container)


def ens_from_address(address) -> str:
    try:
        web3 = context.get_web3_provider()
        ns = ENS.from_web3(web3)
        domain = ns.name(address)
        if domain is None:
            return f"No ENS domain for {address}"
        else:
            return f"The ENS domain for {address} is {domain}"
    except ValueError as valueError:
        return f"Invalid address {address}"
    except Exception as e:
        traceback.print_exc()
        return f"Unable to process address {address}"


def address_from_ens(domain) -> str:
    try:
        web3 = context.get_web3_provider()
        ns = ENS.from_web3(web3)
        address = ns.address(domain)
        if address is None:
            return f"No address for {domain}"
        else:
            return f"The address for {domain} is {address}"
    except Exception as e:
        traceback.print_exc()
        return f"Unable to process domain {domain}"


@dataclass
class TxPayloadForSending(ContainerMixin):
    user_request_status: Literal['success', 'error']
    parsed_user_request: str = ''
    tx: Optional[dict] = None  # from, to, value, data, gas
    is_approval_tx: bool = False
    error_msg: Optional[str] = None
    description: str = ''

    @classmethod
    def from_workflow_result(cls, result: ui_workflows.base.Result):
        return TxPayloadForSending(
            user_request_status=result.status,
            tx=result.tx,
            error_msg=result.error_msg,
            description=result.description
        )

    def container_name(self) -> str:
        return 'display-tx-payload-for-sending-container'

    def container_params(self) -> Dict:
        return dataclass_to_container_params(self)

@error_wrap
@ensure_wallet_connected
def set_ens_text(domain: str, key: str, value: str) ->TxPayloadForSending:
    wallet_chain_id = 1 # TODO: get from context
    wallet_address = context.get_wallet_address()
    user_chat_message_id = context.get_user_chat_message_id()

    params = {
        'domain': domain,
        'key': key,
        'value': value,
    }

    result = ens.ENSSetTextWorkflow(wallet_chain_id, wallet_address, user_chat_message_id, params).run()
    return TxPayloadForSending.from_workflow_result(result)

@error_wrap
@ensure_wallet_connected
def set_ens_primary_name(domain: str) ->TxPayloadForSending:
    wallet_chain_id = 1 # TODO: get from context
    wallet_address = context.get_wallet_address()
    user_chat_message_id = context.get_user_chat_message_id()

    params = {
        'domain': domain,
    }

    result = ens.ENSSetPrimaryNameWorkflow(wallet_chain_id, wallet_address, user_chat_message_id, params).run()
    return TxPayloadForSending.from_workflow_result(result)

@error_wrap
@ensure_wallet_connected
def set_ens_avatar_nft(domain: str, nftContractAddress: str, nftId: str, collectionName: str) ->TxPayloadForSending:
    wallet_chain_id = 1 # TODO: get from context
    wallet_address = context.get_wallet_address()
    user_chat_message_id = context.get_user_chat_message_id()

    params = {
        'domain': domain,
        'nftContractAddress': nftContractAddress,
        'nftId': nftId,
        'collectionName': collectionName
    }

    result = ens.ENSSetAvatarNFTWorkflow(wallet_chain_id, wallet_address, user_chat_message_id, params).run()
    return TxPayloadForSending.from_workflow_result(result)
