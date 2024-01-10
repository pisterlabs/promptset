from flask import Flask, request
import requests
import time
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from threading import Thread
from datetime import datetime
import json
from web3 import Web3
import random

app = Flask(__name__)
base_url = "https://api.assemblyai.com/v2"

headers = {"authorization": "-api"}
client = Client("-ApiKey", "-ApiKey")

#Eth Starts Here
provider = Web3.HTTPProvider('https://eth-sepolia.g.alchemy.com/v2/-ApiKey')
w3 = Web3(provider)

contract_address = Web3.to_checksum_address("-Contract Address")


contract_abi =[
    {
      "inputs": [],
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "id",
          "type": "uint256"
        },
        {
          "internalType": "string",
          "name": "D",
          "type": "string"
        }
      ],
      "name": "addOperation",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "op",
          "type": "address"
        }
      ],
      "name": "addToWhitelist",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getOperations",
      "outputs": [
        {
          "components": [
            {
              "internalType": "uint256",
              "name": "id",
              "type": "uint256"
            },
            {
              "internalType": "string",
              "name": "data",
              "type": "string"
            },
            {
              "internalType": "uint256",
              "name": "status",
              "type": "uint256"
            }
          ],
          "internalType": "struct Contract.OperationData[]",
          "name": "",
          "type": "tuple[]"
        }
      ],
      "stateMutability": "view",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "op",
          "type": "address"
        }
      ],
      "name": "removeFromWhitelist",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "id",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "s",
          "type": "uint256"
        }
      ],
      "name": "setStatus",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ]

contract = w3.eth.contract(address=contract_address, abi=contract_abi)
sender_address = Web3.to_checksum_address("-Account Address")
w3.eth.defaultAccount = sender_address
nonce = w3.eth.get_transaction_count(sender_address)
data = contract.functions.getOperations().build_transaction({
    'gas': 100000,
    'gasPrice': w3.to_wei('20', 'gwei'),
    'nonce': nonce,
})
signed_txn = w3.eth.account.sign_transaction(data, private_key='-PrivateKey')
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
#Eth Ends Here

@app.route("/answer", methods=['GET', 'POST'])
def answer_call():
  """Respond to incoming phone calls with a brief message."""
  # Start our TwiML response
  resp = VoiceResponse()

  # Read a message aloud to the caller
  resp.say(
    "Hello, this is AI the emergency helpline. Please state your name, current location, and describe your emergency situation, Cut the call when you are finished, Your request will be registered once you end the call and we will notify you as your request is registered.",
    voice='alice')
  resp.record(maxLength="120", action="/handle-recording")
  resp.say("Thank you for your message. Goodbye.", voice='alice')
  return str(resp)



@app.route("/handle-recording", methods=['GET', 'POST'])
def handle_recording():
  recording_url = request.form['RecordingUrl']
  data = {"audio_url": recording_url}
  # print(data)
  url = base_url + "/transcript"
  response = requests.post(url, json=data, headers=headers)
  # print(response)
  transcript_id = response.json()['id']


  polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

  
  while True:
    transcription_result = requests.get(polling_endpoint,
                                        headers=headers).json()

    # print(transcribed_text)

    if transcription_result['status'] == 'completed':
      # Get the call sid from the request
      call_sid = request.form['CallSid']

      # Get the call details from Twilio
      call = client.calls(call_sid).fetch()
      caller_number = call.from_formatted

      print(caller_number)
      print("Getting priority info from OpenAI");
      transcription_text = transcription_result['text'];
      print(transcription_text)

      s="a call from :"+caller_number+"the context is :"+transcription_text

         
      rand_12_digit_id = random.randint(100000000000,999999999999)
     
      transaction = contract.functions.addOperation(rand_12_digit_id,s).build_transaction({
             'gas': 2000000,
             'gasPrice': w3.to_wei('50', 'gwei'),
             'nonce': w3.eth.get_transaction_count(sender_address),
      })

      signed_txn = w3.eth.account.sign_transaction(transaction, private_key='-PrivateKey')
      tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

      print(tx_hash)
      print("Transaction Completed Successfully.")
      print("\n")

      print("Transaction Failed!")
      print("\n")

      break

    elif transcription_result['status'] == 'error':
      raise RuntimeError(
        f"Transcription failed: {transcription_result['error']}")

    else:
      time.sleep(3)
  return "OK"


def run():
  app.run(host='0.0.0.0', port=80)


run()


