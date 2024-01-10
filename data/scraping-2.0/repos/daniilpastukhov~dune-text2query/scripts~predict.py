import os

import openai
from dotenv import load_dotenv

load_dotenv('../.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    prompt = """
### Postgres SQL tables, with their properties:
#
# ethereum.transactions(block_time, nonce, index, success, from, to, value, block_number, block_hash, gas_limit, gas_price, gas_used, data, hash, type, access_list, max_fee_per_gas, max_priority_fee_per_gas, priority_fee_per_gas)
# ethereum.blocks(time, number, hash, parent, gas_limit, gas_used, miner, difficulty, total_difficulty, nonce, size, base_fee_per_gas)
# ethereum.logs(block_hash, block_number, block_time, contract_address, topic1, topic2, topic3, topic4, data, tx_hash, index, tx_inde)
# ethereum.traces(block_time, tx_success, success, block_hash, block_number, tx_hash, from, to, value, gas, gas_used, tx_index, trace_address, sub_traces, type, address, code, call_type, can, For, input, output, refund_address)
#
### A query to count users who mint NFT on Ethereum in last 3 months

SELECT"""

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response['choices'][0]['text'].strip())
