import requests
import json
import io
import os
import openai
from web3 import Web3, HTTPProvider

def get_embedding(text):
  res = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=text
  )
  return res

def get_abi(contract_address, apikey):    
    url = f"https://api.etherscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={apikey}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == '1':
        abi = json.loads(data['result'])
        return abi
    else:
        raise Exception(f"Failed to get ABI: {data['message']}")

def get_token_uri(api_key, contract_address, abi, token_id):
    web3 = Web3(HTTPProvider(f'https://eth-mainnet.g.alchemy.com/v2/{api_key}'))
    contract = web3.eth.contract(address=contract_address, abi=abi)

    token_uri = contract.functions.tokenURI(token_id).call()
    return token_uri

def parse_ipfs_url(ipfs_url):
    return ipfs_url.replace('ipfs://', '')

def get_ipfs_content(cid):
    url = f"https://ipfs.io/ipfs/{cid}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to get content: HTTP {response.status_code}")

if __name__ == "__main__":  
    openai.api_key = os.environ["OPENAI_API_KEY"]  
    eth_api_key = os.environ["ETHERSCAN_API_KEY"]
    alc_api_key = os.environ["ALCHEMY_API_KEY"]

    contract_address = ""
    abi = get_abi(contract_address, eth_api_key)
    token_id = 1
    
    ipfs_url = get_token_uri(alc_api_key, contract_address, abi, token_id)
    cid = parse_ipfs_url(ipfs_url)
    content = get_ipfs_content(cid)
    embedding = get_embedding(content)
    # save(embedding, cid)
