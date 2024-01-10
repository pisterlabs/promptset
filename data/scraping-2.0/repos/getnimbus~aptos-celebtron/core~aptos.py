from embedchain import App
from datetime import datetime
from decimal import Decimal
import openai
import requests
import logging

logger = logging.getLogger(__name__)


def get_tx_by_version(version_id: str) -> dict:
    try:
        url = f"https://fullnode.mainnet.aptoslabs.com/v1/transactions/by_version/{version_id}"
        headers = {
            "Accept": "application/json",
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            json_data = response.json()
            return {
                "transaction_fee": f'{str(Decimal(json_data["gas_used"]) * Decimal(json_data["gas_unit_price"]) / 10 ** 8)} APT',
                "success": json_data["success"],
                "vm_status": json_data["vm_status"],
                "sender": json_data["sender"],
                "events": json_data["events"],
                "timestamp": datetime.utcfromtimestamp(int(json_data["timestamp"]) / 1000000.0).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "type": json_data["type"],
            }
        else:
            logger.error("Request failed with status code %d", response.status_code)
            return None
    except Exception as e:
        logger.error(e)
        return None


def get_tx_by_hash(tx_hash: str) -> dict:
    try:
        url = f"https://fullnode.mainnet.aptoslabs.com/v1/transactions/by_hash/{tx_hash}"
        headers = {
            "Accept": "application/json",
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            json_data = response.json()
            return {
                "transaction_fee": f'{str(Decimal(json_data["gas_used"]) * Decimal(json_data["gas_unit_price"]) / 10 ** 8)} APT',
                "success": json_data["success"],
                "vm_status": json_data["vm_status"],
                "sender": json_data["sender"],
                "events": json_data["events"],
                "timestamp": datetime.utcfromtimestamp(int(json_data["timestamp"]) / 1000000.0).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                "type": json_data["type"],
            }
        else:
            logger.error("Request failed with status code %d", response.status_code)
            return None
    except Exception as e:
        logger.error(e)
        return None


def generate_prompt(input_query, context):
    logger.info(context)
    prompt = f"""Use the following pieces of context to answer the query at the end. If you don"t know the answer, just say that you don"t know, don"t try to make up an answer.
        {context}
        Query: {input_query}
        Helpful Answer:
        """
    return prompt


def get_openai_answer(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant. You are given a json object of transaction in Aptos chain. You must explain the transaction in short.",
        },
        {
            "role": "system",
            "content": "The answer must contains datetime, sender/receiver address, gas fee, event breakdown and determine main type of transaction."
        },
        {
            "role": "system",
            "content": '''The transaction has one of following events then this event type is the main transaction type:
              1. If event has including type is "0x54ad3d30af77b60d939ae356e6606de9a4da67583f02b962d2d3f2e481484e90::executor_v1::AirdropEvent" then it must be an airdrop tokens/NFTs transaction.
              2. If event has including type is "0x190d44266241744264b964a37b8f09863167a12d3e70cda39376cfb4e3561e12::liquidity_pool::SwapEvent" then it must be a swap token in liquidity pool. For example if 0x190d44266241744264b964a37b8f09863167a12d3e70cda39376cfb4e3561e12::liquidity_pool::SwapEvent<0xa2eda21a58856fda86451436513b867c97eecb4ba099da5775520e0f7492e852::coin::T, 0x1::aptos_coin::AptosCoin, 0x190d44266241744264b964a37b8f09863167a12d3e70cda39376cfb4e3561e12::curves::Uncorrelated> then 0x1::aptos_coin::AptosCoin is input token and 0x190d44266241744264b964a37b8f09863167a12d3e70cda39376cfb4e3561e12::curves::Uncorrelated is output token.
              3. If event has including type is "0x190d44266241744264b964a37b8f09863167a12d3e70cda39376cfb4e3561e12::liquidity_pool::LiquidityAddedEvent" then it must be an adding liquidity transaction.
              4. If transaction has consist only 2 events "0x1::coin::DepositEvent" and "0x1::coin::WithdrawEvent" then it must be a transfer token transaction.'''
        },
        {
            "role": "user", "content": prompt
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=0,
        top_p=1,
    )
    return response["choices"][0]["message"]["content"]


def ask_question(input_query):
    context = app.retrieve_from_database(input_query)
    prompt = generate_prompt(input_query, context)
    answer = get_openai_answer(prompt)
    return answer


app = App()

app.add("web_page", "https://aptos.dev/concepts/txns-states")

if __name__ == "__main__":
    # AirdropEvent
    # https://explorer.aptoslabs.com/txn/169470665?network=mainnet
    # tx = get_tx_by_version("169470665")

    # TransferEvent
    # https://explorer.aptoslabs.com/txn/169646501?network=mainnet
    # tx = get_tx_by_version("169646501")

    # SwapEvent
    # https://explorer.aptoslabs.com/txn/169674936?network=mainnet
    # tx = get_tx_by_version("169674936")

    # SwapEvent USDC
    # https://explorer.aptoslabs.com/txn/169707501?network=mainnet
    tx = get_tx_by_version("169707501")

    # LiqudityAddedEvent
    # https://explorer.aptoslabs.com/txn/169707427?network=mainnet
    # tx = get_tx_by_version("169707427")

    # debugging tx info
    print(tx)

    # tx explanation
    input_query = f"Explain this transaction: {tx}"
    answer = ask_question(input_query)
    print(answer)
