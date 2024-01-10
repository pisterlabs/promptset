import json
import os
import asyncio
from web3 import Web3
from dotenv import load_dotenv
import openai
import requests
from datetime import datetime, timedelta
import queue
import logging
import time

load_dotenv()

#############
# Creator Wallet  0xA1424c75B7899d49dE57eEF73EabF0e85e093D44
#############

CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
ABI_JSON = json.loads(os.getenv("ABI_JSON"))
openai.api_key = os.getenv('DALLI_KEY')
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
ACCOUNT_ADDRESS = os.getenv("ACCOUNT_ADDRESS")
HTTPProvider_RPC = os.getenv("HTTPProvider_RPC")
PINATA_KEY = os.getenv("PINATA_KEY")
PINATA_URL = os.getenv("PINATA_URL")
receiver_address = '0x000000000000000000000000000000000000dead'
pref_domain = 'https://ipfs.io/ipfs/'
events_data = {}
size = "1024x1024"
cids = []
metas = []
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
tx_queue = queue.Queue()


w3 = Web3(Web3.HTTPProvider(HTTPProvider_RPC))
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI_JSON)


def greet_time(value):
    match value:
        case "night":
            return 'QmaciAWDBQdFWXTiiagSgqZqyoE5RVTNK6XzsArySHcXHq'
        case "morning":
            return 'QmNPXmFv1xiN84Hm2JDrFDnBBXijpDPXxAs5TFWM2Wh2uG'
        case "noon":
            return 'QmR8AjJtKdS7QLB7iKuHJtyM1HXpn6ZU93P7NAKcExeNuy'
        case "evening":
            return 'QmXzsp5L9dZaGtnXmv2vbKhCwUCZoYBpyP5hWa8o5DA2de'
        case _:
            print("WTF")
            return None


def burn_token(token_id):
    max_attempts = 2
    attempt = 0
    tx_receipt = None
    dynamic_max_fee_per_gas = 700000

    while attempt < max_attempts:
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        chain_id = w3.eth.chain_id

        transaction = contract.functions.burnToken(token_id).build_transaction({
            "chainId": chain_id,
            "from": ACCOUNT_ADDRESS,
            'gas': dynamic_max_fee_per_gas,
            'maxFeePerGas': w3.to_wei('2', 'gwei'),
            'maxPriorityFeePerGas': w3.to_wei('1', 'gwei'),
            'nonce': nonce
        })

        signed_txn = w3.eth.account.sign_transaction(
            transaction, private_key=PRIVATE_KEY)
        logger.info(str(
            f'signed_txn !!!! {signed_txn}'))
        send_tx = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        tx_receipt = w3.eth.wait_for_transaction_receipt(send_tx)

        if tx_receipt.status == 1:
            logger.info(str(f'BURN OK !!!'))
            break
        attempt += 1
        logger.info(str(f'ERROR OK !!!'))
        gas_used = int(tx_receipt.gasUsed)
        logger.info(str(f'gas_used : {gas_used}'))
        dynamic_max_fee_per_gas = int(gas_used * 1.2)
        logger.info(
            str(f'dynamic_max_fee_per_gas : {dynamic_max_fee_per_gas}'))

    return tx_receipt


def pinata_upload(filename):

    # Ensure that the API key is available
    if PINATA_KEY is None:
        logger.info(
            str(f'PINATA_KEY is not set. Please check your .env file. !!!'))
        return

    url = f'{PINATA_URL}'
    headers = {
        "accept": "application/json",
        'authorization': f'Bearer {PINATA_KEY}'
    }

    with open(filename, "rb") as file:
        files = {"file": (filename, file, "image/png")}
        response = requests.post(url, files=files, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print('IPFS Hash:', data['IpfsHash'])
        return data['IpfsHash']
    else:
        logger.info(str(f'Failed to upload to IPFS'))
        logger.info(str(f'Status Code: {response.status_code}'))
        logger.info(str(f'Response: {response.text}'))
        # print("Failed to upload to IPFS")
        # print('Status Code:', response.status_code)
        # print('Response:', response.text)
        return None


def add_milliseconds_and_convert_to_unixtime(numbers):
    total_milliseconds = sum(int(str(num)[-8:]) for num in numbers)
    added_seconds = total_milliseconds // 1000

    now = datetime.now()

    new_date = now + timedelta(milliseconds=total_milliseconds)

    return int(new_date.timestamp()), added_seconds

# Idea Ambientiumim && RuFFbuff

def add_uris_to_token(token_id, uris, burnInSeconds, imageURL):
    max_attempts = 2
    attempt = 0
    tx_receipt = None
    dynamic_max_fee_per_gas = 700000

    while attempt < max_attempts:
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        chain_id = w3.eth.chain_id

        transaction = contract.functions.addURIBatch(token_id, uris, burnInSeconds, imageURL).build_transaction({
            "chainId": chain_id,
            "from": ACCOUNT_ADDRESS,
            'gas': dynamic_max_fee_per_gas,
            'maxFeePerGas': w3.to_wei('5', 'gwei'),
            'maxPriorityFeePerGas': w3.to_wei('1', 'gwei'),
            'nonce': nonce
        })

        signed_txn = w3.eth.account.sign_transaction(
            transaction, private_key=PRIVATE_KEY)
        logger.info(str(
            f'signed_txn !!!! {signed_txn} '))
        send_tx = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        tx_receipt = w3.eth.wait_for_transaction_receipt(send_tx)

        if tx_receipt.status == 1:
            logger.info(str(f'URIS Wrote'))
            break
        attempt += 1
        logger.info(str(f'URIS ERROR !!!'))
        gas_used = int(tx_receipt.gasUsed)
        logger.info(str(f'gas_used : {gas_used}'))
        dynamic_max_fee_per_gas = int(gas_used * 1.2)
        logger.info(
            str(f'dynamic_max_fee_per_gas :  {dynamic_max_fee_per_gas}'))
        time.sleep(1)

    return tx_receipt

# created and compiled by Denis Kurochkin (c)2023
# yatadcd@gmail.com

def generate_image_with_dalle(prompt):
    try:
        response = openai.Image.create(
            prompt=prompt, n=1, size=size, quality="standard", model="dall-e-3")
        return response
    except Exception as e:
        logger.info(str(f'An error occurred: {e}'))
        return None


def save_image(image_url, filename):
    response = requests.get(image_url)
    with open(filename, "wb") as f:
        f.write(response.content)
    logger.info(str(f'Image saved as {filename}'))


def create_nft_metadata(response, tockenId, image_cid, file_path, time_of_day, expiration_time):
    image_url = f"{pref_domain}{image_cid}"
    metadata = {
        "name": f"DynNFT {tockenId}",
        "description": "This Dynamic NFT for a chailink hackathon",
        "image": image_url,
        "attributes": [
            {"trait_type": "Token ID", "value": response.get("tokenId")},
            {"trait_type": "Time_of_day", "value": time_of_day},
            {"trait_type": "Animal", "value": response.get("animal")},
            {"trait_type": "Name", "value": response.get("name")},
            {"trait_type": "Country", "value": response.get("country")},
            {"trait_type": "Style", "value": response.get("style")},
            {"display_type": "boost_number", "trait_type": "VRF Random",
                "value": response.get("randomNumbers")},
            {"display_type": "date", "trait_type": "Deadline",
                "value": expiration_time}
        ]
    }
    try:
        with open(file_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        logger.info(str(f'Metadata saved to {file_path}'))
    except Exception as e:
        logger.info(str(f'An error occurred: {e}'))


async def handle_event(event):
    event_name = event.event
    event_args = event.args

    if event_name == 'WeatherNFTMinted':
        token_id = event_args['tokenId']
        events_data[token_id] = {
            'tokenId': token_id,
            'animal': event_args['animal'],
            'name': event_args['name'],
            'country': event_args['country'],
            'style': event_args['style']
        }
    elif event_name == 'RandomnessFulfilled':
        token_id = event_args['tokenId']
        if token_id in events_data:
            events_data[token_id].update({
                'randomNumbers': event_args['randomNumbers']
            })
            logger.info(
                str(f'Combined data for token {token_id}: {events_data[token_id]}'))
            response_data = events_data[token_id]
            style = response_data['style']
            animal = response_data['animal']
            name = response_data['name']
            country = response_data['country']
            image_name_file = response_data['randomNumbers'][0]
            json_name_file = response_data['randomNumbers'][1]
            randtime = [image_name_file, json_name_file]
            expiration_time, added_seconds = add_milliseconds_and_convert_to_unixtime(
                randtime)
            logger.info(str(f'expiration_time: {expiration_time}'))
            logger.info(str(f'add time: {added_seconds}'))
            # added_seconds = 300
            times_of_day_prompts = {
                "night": f"Draw image: Time of day: Night, Place:{country}, Mood: Fanny, Character: {animal}, Style: {style}",
                "morning": f"Draw image: Time of day: Morning, Place:{country}, Mood: Fanny, Character: {animal}, Style: {style}",
                "noon": f"Draw image: Time of day: Noon, Place:{country}, Mood: Fanny, Character: {animal}, Style: {style}",
                "evening": f"Draw image: Time of day: Evening, Place:{country}, Mood: Fanny, Character: {animal}, Style: {style}"
            }
            for time_of_day, prompt in times_of_day_prompts.items():
                logger.info(str(f'Generating image for {time_of_day}...'))
                response = generate_image_with_dalle(prompt)
                if response and response['data']:
                    image_url = response['data'][0]['url']
                    base_filename = f"{image_name_file:-10}_{time_of_day}.png"
                    json_filename = f"{json_name_file:-10}_{time_of_day}.json"
                    save_image(image_url, base_filename)
                    cid = pinata_upload(base_filename)
                    cids.append(cid)
                    os.remove(base_filename)
                    create_nft_metadata(
                        response_data, token_id, cid, json_filename, time_of_day, expiration_time)
                    cid = pinata_upload(json_filename)
                    metas.append(cid)
                    os.remove(json_filename)
                if response == None:
                    base_filename = f"{image_name_file:-10}_{time_of_day}.png"
                    json_filename = f"{json_name_file:-10}_{time_of_day}.json"
                    cid = greet_time(time_of_day)
                    cids.append(cid)
                    create_nft_metadata(
                        response_data, token_id, cid, json_filename, time_of_day, expiration_time)
                    cid = pinata_upload(json_filename)
                    metas.append(cid)
                    os.remove(json_filename)

            prepare_img = [
                pref_domain + cid for cid in cids]

            prepare_img = ','.join(prepare_img)

            logger.info(str(f'prepare_img : , {prepare_img}'))

            prepare_metas = [
                pref_domain + meta for meta in metas]

            logger.info(str(f'prepare_metas : , {prepare_metas}'))
            tx_queue.put_nowait(
                {'func': 'add_uris', 'token_id': token_id,  'prepare_metas': prepare_metas, 'expiration_time': added_seconds, 'imageURL': prepare_img})

            cids.clear()
            metas.clear()

            del events_data[token_id]


async def log_loop(event_filter, poll_interval):
    while True:
        for event in event_filter.get_new_entries():
            await handle_event(event)
        await asyncio.sleep(poll_interval)


async def log_time():
    while True:
        current_time = datetime.now()
        logger.info(str(f'TIME : {current_time}'))
        getAllTokenBurnTimes = contract.functions.getAllTokenBurnTimes().call()

        if len(getAllTokenBurnTimes[0]) == len(getAllTokenBurnTimes[1]) == len(getAllTokenBurnTimes[2]):
            for i in range(len(getAllTokenBurnTimes[0])):
                unix_time = datetime.fromtimestamp(getAllTokenBurnTimes[2][i])
                if unix_time < current_time:
                    time_status = "PAST"
                    tx_queue.put_nowait(
                        {'func': 'burn', 'wallet': getAllTokenBurnTimes[0][i],  'token_id': getAllTokenBurnTimes[1][i], 'time': getAllTokenBurnTimes[2][i]})
                else:
                    time_status = "FUTURE"

                logger.info(str(
                    f"wallet : {getAllTokenBurnTimes[0][i]},  token_id: {getAllTokenBurnTimes[1][i]} , time:{getAllTokenBurnTimes[2][i]}, {time_status}"))
        else:
            logger.info("error.")
        await asyncio.sleep(60)


async def process_tx_queue():
    while True:
        queue_size = tx_queue.qsize()
        logger.info(str(f'Count queues : {queue_size}'))
        if not tx_queue.empty():
            tx_data = tx_queue.get_nowait()
            logger.info(str(f'####  TX: ... #### {tx_data}'))
            if tx_data['func'] == 'burn':
                token_id = tx_data['token_id']
                holder_address = tx_data['wallet']
                time = tx_data['time']
                if time == 0:
                    logger.info(str(f'####  DROP #### {holder_address}'))
                else:
                    logger.info(
                        str(f'####  BURN starting ... #### {token_id}'))
                    burn_token(token_id)

            elif tx_data['func'] == 'add_uris':
                logger.info(str(f'####  URIS writting ... #### '))
                token_id = tx_data['token_id']
                prepare_metas = tx_data['prepare_metas']
                expiration_time = tx_data['expiration_time']
                prepare_img = tx_data['imageURL']
                add_uris_to_token(token_id, prepare_metas, expiration_time, prepare_img)

        await asyncio.sleep(2)


async def main():
    logger.info(CONTRACT_ADDRESS)
    w3 = Web3(Web3.HTTPProvider(HTTPProvider_RPC))
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI_JSON)
    logger.info(
        str(f'Listening for events from contract at {CONTRACT_ADDRESS}'))

    weather_nft_minted_filter = contract.events.WeatherNFTMinted.create_filter(
        fromBlock='latest')
    randomness_fulfilled_filter = contract.events.RandomnessFulfilled.create_filter(
        fromBlock='latest')

    await asyncio.gather(
        log_loop(weather_nft_minted_filter, 1),
        log_loop(randomness_fulfilled_filter, 1),
        log_time(),
        process_tx_queue()
    )

if __name__ == '__main__':
    asyncio.run(main())
