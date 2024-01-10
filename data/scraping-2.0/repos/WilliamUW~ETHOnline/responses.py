from io import BytesIO
import random
import time
import json
import discord
import face_recognition
import skimage
import requests
from PIL import Image
import numpy as np
from masknetwork import getETHAddressUsingMask

from web3storage import uploadImageToIPFS

step = 0
name = "Name"
description = "Description"
chain = "Chain"
registerFlow = False
connectFlow = False
hederaFlow = False
flowFlow = False
flowTransactionFlow = False
hederaTransactionFlow = False

import requests

from dotenv import dotenv_values
import os
from twilio.rest import Client

import w3storage

w3 = w3storage.API(token="w3-api-token")


config = dotenv_values(".env")  # Load .env file

# Access the API keys
openai_api_key = config["OPENAI_API_KEY"]
discord_token = config["DISCORD_TOKEN"]
account_sid = config["TWILIO_ACCOUNT_SID"]
auth_token = config["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

# Use the API keys in your code
print(openai_api_key)
print(discord_token)

hedera_to_encoding = {}

flow_to_encoding = {}

wallet_to_phone = {"donaldtrump.eth": "4168807375", "barackobama.eth": "4168807375"}

wallet_to_encoding = {
    "donaldtrump.eth": [
        -0.14075439,
        0.11751342,
        0.00939059,
        0.02285596,
        -0.11728013,
        -0.03425988,
        0.04682633,
        -0.17926697,
        0.08297635,
        -0.08466988,
        0.17541729,
        -0.08900289,
        -0.36073408,
        -0.08471457,
        0.00179974,
        0.12912574,
        -0.15396717,
        -0.15580328,
        -0.2022332,
        -0.12501474,
        0.03356066,
        -0.00221383,
        -0.01778279,
        -0.04045977,
        -0.11586375,
        -0.22829586,
        -0.05460431,
        -0.13751653,
        0.01975143,
        -0.09269837,
        0.06132156,
        -0.01824741,
        -0.22651678,
        -0.12700555,
        -0.00732809,
        -0.01082183,
        -0.10614757,
        -0.07196499,
        0.182786,
        -0.02742191,
        -0.15820941,
        -0.0256714,
        0.02479331,
        0.20706898,
        0.21993566,
        0.01965954,
        -0.01328283,
        -0.15616775,
        0.10175291,
        -0.2906104,
        -0.00673473,
        0.14541376,
        0.15498871,
        0.11310868,
        0.07136281,
        -0.1229042,
        0.04115922,
        0.14180177,
        -0.211348,
        0.07096414,
        0.07627963,
        -0.15285313,
        -0.03646907,
        -0.04525581,
        0.11244542,
        0.07555193,
        -0.00781911,
        -0.11803017,
        0.28472951,
        -0.17995432,
        -0.13740472,
        0.06245572,
        -0.0253733,
        -0.18989277,
        -0.32309234,
        -0.04631754,
        0.36912838,
        0.18131258,
        -0.19008221,
        -0.08479589,
        -0.10575131,
        -0.02596735,
        0.04534491,
        0.0059922,
        -0.06032074,
        -0.14000094,
        -0.09736186,
        0.02257752,
        0.24148701,
        -0.09898039,
        -0.02243083,
        0.21237141,
        0.05141855,
        -0.07680816,
        0.04236894,
        0.01287101,
        -0.10867499,
        -0.02760276,
        -0.11733112,
        -0.08913676,
        0.064026,
        -0.11971642,
        0.03425536,
        0.1284885,
        -0.14563903,
        0.1710926,
        -0.01813834,
        -0.05552238,
        -0.06981698,
        -0.11856341,
        -0.00665481,
        0.07988882,
        0.18859179,
        -0.16160974,
        0.26198894,
        0.22163773,
        -0.04114643,
        0.09990032,
        0.00601859,
        0.03312224,
        -0.05141105,
        -0.06921807,
        -0.22446799,
        -0.17365944,
        0.03190631,
        0.0351162,
        0.01121672,
        0.04591377,
    ],
    "barackobama.eth": [
        -0.0914344,
        0.13086095,
        0.01314385,
        -0.05788445,
        0.01628965,
        0.00041327,
        -0.08469851,
        -0.09900524,
        0.17989591,
        -0.10539678,
        0.24560224,
        0.08059315,
        -0.2161147,
        -0.13486721,
        0.04742461,
        0.12056788,
        -0.16367513,
        -0.07826022,
        -0.1122469,
        -0.10610124,
        0.03652948,
        0.00634994,
        0.10533702,
        0.04300565,
        -0.12117673,
        -0.33629149,
        -0.06974643,
        -0.18218073,
        -0.00158545,
        -0.1120832,
        -0.09656743,
        -0.02059199,
        -0.18194009,
        -0.1091411,
        0.02073221,
        -0.02022129,
        0.00240957,
        -0.00374015,
        0.20474017,
        0.0282058,
        -0.11632427,
        0.09632833,
        0.01547976,
        0.21318354,
        0.28629938,
        0.07692298,
        -0.01180618,
        -0.09913055,
        0.10386178,
        -0.21633516,
        0.07274053,
        0.14290063,
        0.08237933,
        0.04238797,
        0.09769628,
        -0.18852283,
        0.00360183,
        0.08834425,
        -0.14143489,
        0.00837216,
        0.0078872,
        -0.08102693,
        -0.04035496,
        0.0387958,
        0.20594732,
        0.09965956,
        -0.1229291,
        -0.05094442,
        0.13211268,
        -0.02900139,
        0.02445153,
        0.02434404,
        -0.18431334,
        -0.20063369,
        -0.22774039,
        0.09293823,
        0.37345198,
        0.19359806,
        -0.2088118,
        0.01955765,
        -0.19599999,
        0.02415315,
        0.06105619,
        0.00819598,
        -0.07174452,
        -0.13538505,
        -0.04118638,
        0.05282182,
        0.0822657,
        0.03208514,
        -0.04098899,
        0.21506976,
        -0.03382806,
        0.06236776,
        0.01853621,
        0.05682226,
        -0.15838756,
        -0.03170495,
        -0.16015227,
        -0.06845063,
        0.01404157,
        -0.04203653,
        0.03085331,
        0.14781639,
        -0.23243298,
        0.05921936,
        0.00418688,
        -0.04666766,
        0.0222913,
        0.07022521,
        -0.02721735,
        -0.03373824,
        0.05814214,
        -0.23816805,
        0.24889056,
        0.23403469,
        0.02495461,
        0.17327937,
        0.07225873,
        0.03394287,
        -0.01637957,
        -0.02267808,
        -0.18229848,
        -0.06459411,
        0.06046797,
        0.0755232,
        0.0852315,
        0.00671965,
    ],
}

william_encoding = [
    -0.08489514,
    0.16629207,
    -0.00341852,
    -0.0089607,
    -0.07309289,
    -0.0365204,
    -0.07142913,
    -0.09204409,
    0.13079758,
    -0.06249057,
    0.18775795,
    -0.07102738,
    -0.24563771,
    -0.12483668,
    -0.07561519,
    0.21899216,
    -0.19840415,
    -0.16191454,
    -0.06565362,
    0.01267712,
    0.12735552,
    0.0060061,
    -0.03330318,
    0.0601213,
    -0.09948364,
    -0.37896919,
    -0.04892989,
    -0.08004473,
    0.0248933,
    -0.04926632,
    -0.06370206,
    -0.02361014,
    -0.20606354,
    -0.07222925,
    0.02395701,
    0.05067029,
    -0.0600784,
    -0.03047261,
    0.1981484,
    0.0334802,
    -0.25515872,
    0.03835013,
    0.05569423,
    0.26776901,
    0.21762674,
    0.07363212,
    0.04883651,
    -0.15051921,
    0.11447029,
    -0.18776159,
    0.03032518,
    0.19864576,
    0.0858774,
    0.08481935,
    0.00737492,
    -0.13686721,
    -0.01707979,
    0.15946123,
    -0.14788455,
    0.03676679,
    0.05411533,
    -0.10939586,
    -0.02040504,
    -0.06258389,
    0.28817016,
    0.02403246,
    -0.11974652,
    -0.16277888,
    0.17104167,
    -0.15995584,
    -0.08980826,
    0.01079151,
    -0.09745924,
    -0.17857276,
    -0.33928576,
    0.04938306,
    0.37440464,
    0.09512523,
    -0.197273,
    0.11345024,
    -0.04765933,
    -0.02723197,
    0.14252216,
    0.1973733,
    -0.00534958,
    0.0197605,
    -0.0809767,
    -0.01989206,
    0.24185868,
    -0.08026418,
    0.02053601,
    0.22454603,
    -0.01001545,
    0.07074997,
    0.02581748,
    -0.02406856,
    -0.10836311,
    0.02895864,
    -0.10457729,
    -0.04334836,
    0.04199492,
    -0.02356306,
    0.01468095,
    0.09057769,
    -0.11326458,
    0.08015896,
    0.01140614,
    0.05092007,
    0.02000132,
    -0.01679315,
    -0.09414355,
    -0.1160293,
    0.12748165,
    -0.19033121,
    0.27831239,
    0.1562001,
    0.06773084,
    0.10076032,
    0.16593421,
    0.07828102,
    0.02960114,
    -0.01275891,
    -0.21681403,
    0.00667689,
    0.14095992,
    -0.04186418,
    0.09741175,
    -0.00806305,
]

import openai


def generate_text(prompt):
    # Set up your OpenAI API credentials
    openai.api_key = openai_api_key

    # Define the model and parameters
    model = "text-davinci-003"
    max_tokens = 300  # Maximum number of tokens in the generated response

    prompt = "Concisely answer the following question: " + prompt

    # Generate text using the prompt
    response = openai.Completion.create(
        engine=model, prompt=prompt, max_tokens=max_tokens
    )

    # Extract the generated text from the API response
    generated_text = response.choices[0].text.strip()

    return generated_text


welcome_message = """Hi - welcome to FaceLink! How can I help?
        
Type "Register" to get started!
Type "Connect" to connect with anyone with an image of their face!
"""


def get_image_from_url(image_url: str):
    print(image_url)

    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open and convert the image to a NumPy array
        rgba_image = Image.open(BytesIO(response.content))
        img = rgba_image.convert("RGB")
    else:
        print("Failed to retrieve the image. Status code:", response.status_code)

    print(img)

    return img


def get_response(message_string: str, message: any, is_private: any) -> str:
    global step
    global name
    global description
    global image_url
    global chain
    global registerFlow
    global connectFlow
    global hederaFlow
    global hederaTransactionFlow
    global flowFlow
    global flowTransactionFlow

    p_message = message_string.lower()

    if p_message == "hi" or p_message == "hello":
        return welcome_message

    if p_message == "!help":
        return welcome_message

    if p_message == "register":
        registerFlow = True
        return """Let's get you registered! Please upload an image of yourself.
        
Please type your Ethereum wallet address in the same message.

E.g. "0x0E5d299236647563649526cfa25c39d6848101f5" or "williamw.eth"
"""

    if p_message == "hedera":
        registerFlow = True
        return """Let's connect your face to Hedera! Please upload an image of yourself.
        
Please type your preferred Hedera address in the same message you attach your image."""

    if p_message == "hedera account":
        hederaFlow = True
        return """Forgot your account ID? Let's connect your face to Hedera to easily view your account information! Please upload an image of yourself.
        
"""
    if p_message == "hedera transaction":
        hederaTransactionFlow = True
        return (
            f"Please upload an image of the person you want to send a transaction to!"
        )

    if p_message == "flow":
        registerFlow = True
        return """Let's connect your face to FLOW! Please upload an image of yourself.
        
Please type your preferred Flow address in the same message you attach your image."""

    if p_message == "flow transaction":
        flowTransactionFlow = True
        return (
            f"Please upload an image of the person you want to send a transaction to!"
        )

    if registerFlow:
        print(message)
        if message_string == "":
            return "Sorry, I didn't get a wallet address, please try again!"
        if not (
            (message_string[:2] == "0x" and len(message_string) == 42)
            or (message_string[-4:] == ".eth")
        ):
            # not valid address
            return "Sorry, I didn't get a wallet address, please try again!"
        if message.attachments:
            address = getETHAddressUsingMask(message_string)

            image_url = message.attachments[0].url

            img = get_image_from_url(image_url)

            image_array = np.array(img)

            face_locations = face_recognition.face_locations(image_array)

            print("face_locations: ", face_locations)

            face_encoding = face_recognition.face_encodings(image_array)[0]

            discord_username = str(message.author)

            if len(p_message) > 0:
                discord_username = message_string
            print("User: ", discord_username)
            print("Encoding: ", face_encoding)

            wallet_to_encoding[discord_username] = face_encoding

            # Assuming 'img' is a Discord.py image object
            face_locations = face_locations[0]
            X, Y, W, H = face_locations  # Unpack the values

            # Crop the image
            cropped_image = img.crop((X, Y, X + W, Y + H))

            registerFlow = False

            # message.author.send(file=discord.File(bytearray(cropped_image.read())))

            return f"""Registration Successful!

You have linked your face: {uploadImageToIPFS(image_url)} (Stored on IPFS)

To your wallet address: {discord_username}

With the following encoding: {str(face_encoding)[:200]}... [2681 more characters]
"""

        else:
            return "Sorry, I didn't get an image, please try again!"

    if p_message == "connect":
        connectFlow = True
        print(wallet_to_encoding)
        return f"Let's get you connected! Please upload an image of the person you want to contact. \n\n In the same message, please write the message you want to send to them!"

    if connectFlow or hederaFlow or hederaTransactionFlow or p_message == "":
        print(message)

        if message.attachments:
            print("image")

            image_url = message.attachments[0].url

            img = get_image_from_url(image_url)

            image_array = np.array(img)

            known_faces = list(wallet_to_encoding.values())
            known_faces_names = list(wallet_to_encoding.keys())

            print(known_faces)

            unknown_face_encoding = face_recognition.face_encodings(image_array)[0]

            results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

            print(known_faces_names)
            print(results)

            try:
                index = results.index(True)
                recipient = known_faces_names[index]

                print("Name: ", recipient)
                print("The index of the first True element is:", index)

                discordAuthor = str(message.author)

                if (
                    recipient in wallet_to_phone
                    and message_string != ""
                    and recipient.isnumeric()
                    and len(recipient) == 10
                ):
                    twiliomessage = client.messages.create(
                        body=f"Hi, FaceLink here - {discordAuthor} wants to reach out to you! \n\n Their Discord username is {discordAuthor}. \n\n Their message for you is: {str(message_string)}",
                        from_="+12295750071",
                        to=f"+1{wallet_to_phone[recipient]}",
                    )

                    print(twiliomessage.sid)

                return f"""Face Recognition Successful! 
                
Image: {image_url} 

Request Recipient: {recipient} (This can be hidden based on user privacy preferences)

Reach Out Message: {str(message_string)}

With the following encoding: {str(unknown_face_encoding)[:200]}... [2727 more characters]
"""
            except ValueError:
                print("No True element found in the list.")

            connectFlow = False
            hederaFlow = False
            return "Sorry, no match found :("

        else:
            return "Sorry, I didn't get an image, please try again!"

    else:
        return generate_text(message)
