import random
import requests
import threading
import time
import json
from discord_webhook import DiscordWebhook, DiscordEmbed
from datetime import datetime, timedelta
import tls_client
import string
import hashlib
from uuid import uuid4
import faker
from openai import OpenAI
import pick
import csv
lock = threading.Lock()

def modify_account(account_index, new_data):
    # Acquire the lock
    with lock:
        # Read the file
        with open("accounts.json", "r") as file:
            accounts = json.load(file)

        # Modify the account
        accounts["accounts"][account_index] = new_data

        # Write the changes back to the file
        with open("accounts.json", "w") as file:
            json.dump(accounts, file, indent=4)

aiAnswers = {}

class GoatScript:
    global aiAnswers

    def __init__(self, index, dropId, card, accountData):
        self.card = card
        self.cardName = card[0]
        self.cardType = card[1]
        self.cardNumber = card[2]
        self.cardExpMonth = card[3]
        self.cardExpYear = card[4]
        self.cardCvv = card[5]
        self.index = index
        self.accountData = accountData
        self.pxApiKey = accountData["pxApiKey"]

        self.dropId = dropId
        self.format_string = "%Y-%m-%dT%H:%M:%S.%fZ"
        self.proxyList = accountData["proxyList"]
        self.key = (self.encode(self.pxApiKey))
        self.trackingData = ""
        self.session = tls_client.Session(
            client_identifier="safari_ios_16_0",
            random_tls_extension_order=True
        )
        self.headers = {
            'host': 'www.goat.com',
            'accept': 'application/json',
            'authorization': 'Token token=""',
            'accept-language': 'en-US,en;q=0.9',
            'x-px-authorization': "",
            'user-agent': 'GOAT/2.66.5 (iPhone; iOS 17.0.3; Scale/3.00) Locale/en',
            'connection': 'keep-alive',
            'content-type': 'application/x-www-form-urlencoded',
        }
        
        # with open("accounts.json","r") as accounts:
        if len(self.accountData["accounts"]) <= index:
            self.account = {
                "email": None,
                "password": None,
                "username": None,
                "userId": None,
                "authToken": None,
                "deviceHash": None,
                "deviceId": None,
                "proxy": self.proxies(),
                "index": index,
                "userFirstName": None,
                "userLastName": None,
                "addressId": None,
                "billingId": None,
            }
            self.email = None
            self.password = None
            self.username = None
            self.userId = None
            self.authToken = None
            self.deviceHash = None
            self.deviceId = None
            self.addressId = None
            self.billingId = None

        else:
            self.account = self.accountData["accounts"][index]
            self.email = self.account["email"]
            self.password = self.account["password"]
            self.username = self.account["username"]
            self.userId = self.account["userId"]
            self.authToken = self.account["authToken"]
            self.deviceHash = self.account["deviceHash"]
            self.deviceId = self.account["deviceId"]
            self.addressId = self.account["addressId"]
            self.billingId = self.account["billingId"]
            self.account["index"] = self.index
            self.account["proxy"] = self.proxies()
        
        # self.account = {
        #     "email": self.email,
        #     "password": self.password,
        #     "username": self.username,
        #     "userId": self.userId,
        #     "authToken": self.authToken,
        #     "deviceHash": self.deviceHash,
        #     "deviceId": self.deviceId,
        #     "proxy": self.proxies(),
        #     "index": index,
        #     "userFirstName": None,
        #     "userLastName": None,
        #     "addressId": self.addressId,
        #     "billingId": self.billingId,
        # }

        self.client = OpenAI(api_key="")
        self.id = str(uuid4())[0:8]

        self.original_post = self.session.post

    def new_post(self, *args, **kwargs):
        self.headers["x-emb-st"] = str(round(datetime.timestamp(datetime.now()) * 1000))
        self.headers["x-emb-id"] = ''.join(random.choices(string.digits + 'ABCDEF', k=32)) 

        return self.original_post(*args, **kwargs)

    def sendWebhook(self, title, description, color, image, price, username, size, shoeName):
        webhookUrl = ""
        webhook = DiscordWebhook(url=webhookUrl)
        embed = DiscordEmbed(title=title, description=description, color=color)
        embed.set_thumbnail(url=image)
        
        embed.add_embed_field(name="Item", value=shoeName, inline=False)
        embed.add_embed_field(name="Price", value="$"+price, inline=True)
        embed.add_embed_field(name="Size", value=size, inline=True)
        embed.add_embed_field(name="Account", value=username, inline=False)
        embed.set_footer(text="__ v0.1.2 | GoatBlackFriday")

        webhook.add_embed(embed)
        webhook.execute()
        
    def print(self, text, color = None):
        if color == "green":
            color = "\033[92m"
        elif color == "red":
            color = "\033[91m"
        elif color == "yellow":
            color = "\033[93m"
        elif color == "blue":
            color = "\033[94m"
        elif color == None:
            color = ""
        greyColor = "\033[90m"
        print(f"{greyColor}{self.username} - {color}{text}\033[0m")
        
        # if "ENQUEUED" not in text:
        with(open("logs.txt", "a")) as f:
            f.write(f"{self.id} - {text}\n")

    def proxies(self):
        # get random proxy from proxy list and format into dict
        proxy = random.choice(self.proxyList)
        # proxy = self.proxyList[self.index]
        proxies = {
            "http": f"http://{proxy}",
            "https": f"http://{proxy}"
        }
        return proxies

    def encode(self, input_string, shift=3):
        encoded = ''
        for char in input_string:
            encoded += chr(ord(char) + shift)
        return encoded
    
    def genPx(self, blocked = False):
        if "pxCookie" not in self.account or blocked:
        # if False:
            # proxy = self.proxyList[self.index]
            proxy = random.choice(self.proxyList)

            splitProxy = proxy.split(":")
            # reformat to http://user:pass@ip:port
            proxy = f"http://{splitProxy[2]}:{splitProxy[3]}@{splitProxy[0]}:{splitProxy[1]}"

            self.print("Generating PX...", "yellow")
            
            request = requests.post("https://api.parallaxsystems.io/gen", json={
                "auth": self.key,
                "site": "goat",
                "proxyregion": "us",
                "region": "com",
                "proxy": proxy
            })
            data = request.json()
            if data["error"]:
                self.print(data)
                self.print("Error generating px!", "red")
                self.genPx()
            else:
                trackingData = data["data"]
                pxCookie = data["cookie"].split("px3=")[1]
                pxVid = data["vid"]
                pxCts = data["cts"]
                secHeader = data["secHeader"]
                userAgent = data["UserAgent"]
                self.session.cookies.update({
                    "_pxvid": pxVid,
                    "pxcts": pxCts,
                    "currency": "USD",
                    "country": "US",
                    "guestCheckoutCohort": "18"
                })
                try:
                    with open("accounts.json","r") as accounts:
                        currentData = json.load(accounts)
                        currentData["accounts"][self.index]["pxCookie"] = pxCookie
                        currentData["accounts"][self.index]["proxy"] = proxy

                    modify_account(self.index, currentData["accounts"][self.index])
                except:
                    pass

                return {
                    "pxCookie": pxCookie,
                    "proxy": proxy
                }
            
            
        else:
            oldFormatProxy = self.account["proxy"]["http"].split("//")[1]
            splitProxy = oldFormatProxy.split(":")
            # # reformat to http://user:pass@ip:port
            proxy = f"http://{splitProxy[2]}:{splitProxy[3]}@{splitProxy[0]}:{splitProxy[1]}"
            return {
                "pxCookie": self.account["pxCookie"],
                "proxy": proxy
            }
    
    def getSession(self):
        self.print(f"Getting session for {self.email}...", "yellow")

        cookieData = self.genPx()
        cookie = cookieData["pxCookie"]
        reqProxy = {
            "http": cookieData["proxy"],
            "https": cookieData["proxy"]
        }
        # get time in ms
        now = datetime.now()
        timestamp = round(datetime.timestamp(now) * 1000)

        self.headers["x-emb-st"] = str(timestamp)
        self.headers["x-px-authorization"] = f"3:{cookie}"
        self.headers["authorization"] = f"Token token=\"{self.authToken}\""

        req = self.session.get(f"https://www.goat.com/api/v1/users/me", headers=self.headers).json()
        count = 0
        while "id" not in req:
            count += 1
            if count > 5:
                self.print("Error getting session [FATAL].", "red")
                exit()
        

            self.print("Error getting session! Refreshing Proxy... - ", "red")

            cookieData = self.genPx(True)
            if "px3=" not in cookieData["pxCookie"]:
                cookie = cookieData["pxCookie"]
            else:
                cookie = cookieData["pxCookie"].split("px3=")[1]
            reqProxy = {
                "http": cookieData["proxy"],
                "https": cookieData["proxy"]
            }
            
            self.headers["x-px-authorization"] = f"3:{cookie}"

            req = self.session.get(f"https://www.goat.com/api/v1/users/me", headers=self.headers).json()

        if req["id"] == self.userId:
            self.print("Session is valid!", "green")

            self.userFirstName = req["name"].split(" ")[0]
            self.userLastName = req["name"].split(" ")[1]
            self.account["userFirstName"] = self.userFirstName
            self.account["userLastName"] = self.userLastName
        else:
            self.print("Session is invalid!", "red")
            exit()

        # self.print("Binding Device...", "blue")

        # data = {
        #     'device[devicePlatform]': 'iOS',
        #     'device[deviceOs]': 'iOS 17.0.3',
        #     'device[currentIp]': reqProxy["http"].split("@")[1].split(":")[0],
        #     'device[sandbox]': '0',
        #     'device[deviceHardware]': 'iPhone14,5',
        #     'device[deviceToken]': self.deviceHash,
        #     'device[deviceIdentifier]': self.deviceId.upper(),
        #     'device[advertisingId]': '00000000-0000-0000-0000-000000000000',
        # }

        # self.new_post('https://www.goat.com/api/v1/devices', headers=self.headers, data=data)

        self.print("Entering BF...")

        data = {"campaign":2}
        response = self.new_post("https://www.goat.com/api/v1/community_users/user-entered-campaign-landing", json=data, headers=self.headers)
        if response.status_code != 200:
            self.print("Error entering BF!", "red")
            self.print(response.text)
    
        else:
            self.print("Entered BF!", "green")

    def onboard(self):
        email = self.account["email"]
        password = self.account["password"]
        username = self.account["username"]
        userId = self.account["userId"]
        authToken = self.account["authToken"]
        deviceHash = self.account["deviceHash"]
        deviceId = self.account["deviceId"]
        reqProxy = self.account["proxy"]
        index = self.account["index"]
        userFirstName = self.account["userFirstName"]
        userLastName = self.account["userLastName"]
        addressId = self.account["addressId"]
        billingId = self.account["billingId"]

        self.print("Accepting TOS...", "blue")

        data = '{"campaign":2}'

        response = self.new_post(
            'https://www.goat.com/api/v1/community_users/accept-terms',
            headers=self.headers,
            data=data,
            proxy=reqProxy,
        )

        while response.status_code != 200:
            self.print("Error accepting TOS! Refreshing Proxy...", "red")
            time.sleep(5)
            cookieData = self.genPx(True)
            if "px3=" not in cookieData["pxCookie"]:
                cookie = cookieData["pxCookie"]
            else:
                cookie = cookieData["pxCookie"].split("px3=")[1]
            reqProxy = {
                "http": cookieData["proxy"],
                "https": cookieData["proxy"]
            }
            
            self.headers["x-px-authorization"] = f"3:{cookie}"

            response = self.new_post(
                'https://www.goat.com/api/v1/community_users/accept-terms',
                headers=self.headers,
                data=data,
                proxy=reqProxy,
            )

        if response.json()["userCampaignStates"][0]["campaignState"] == "CAMPAIGN_STATE_TERMS_ACCEPTED":
            self.print("TOS Accepted!")
        
        self.print("Verifying Location...", "yellow")

        data = {"locationCoordinates":{"longitude":0,"latitude":0},"countryCode":"US"}

        # change up location by 0.1 miles on each
        data["locationCoordinates"]["longitude"] += 0.0001 * index
        data["locationCoordinates"]["latitude"] += 0.0001 * index

        response = self.new_post("https://www.goat.com/api/v1/consumer-segment-access/validate-verified-location", json=data, headers=self.headers).json()

        if response["hasVerifiedLocation"]:
            self.print("Location Verified!", "green")

        self.print("Adding location to profile...", "blue")

        data = {"city":"","country":"US","state":"CA"}

        response = self.new_post("https://www.goat.com/api/v1/community_users/update-user-profile", json=data, headers=self.headers).json()

        self.print("Completing BF Onboarding...", "blue")

        data = {"campaign":2}
        response = self.new_post("https://www.goat.com/api/v1/community_users/complete-onboarding", json=data, headers=self.headers).json()

        if response["userCampaignStates"][0]["campaignState"] == "CAMPAIGN_STATE_ONBOARDED":
            self.print("BF Onboarding Complete!", "green")
        else:
            self.print("Error completing BF Onboarding!", "red")
            self.print(response)

        
            

    def getTickets(self):
        # create needed variables
        email = self.account["email"]
        password = self.account["password"]
        username = self.account["username"]
        userId = self.account["userId"]
        authToken = self.account["authToken"]
        deviceHash = self.account["deviceHash"]
        deviceId = self.account["deviceId"]
        reqProxy = self.account["proxy"]
        index = self.account["index"]
        userFirstName = self.account["userFirstName"]
        userLastName = self.account["userLastName"]
        addressId = self.account["addressId"]
        billingId = self.account["billingId"]
        

        self.print("Finding Trivia...", "blue")
        
        data = {
            "salesChannels": [
                1
            ],
            "recordTypes": [
                6
            ],
            "pageLimit": "75",
            "eventTimeAfter": "2023-11-15T01:10:38.601655006Z",
            "eventTimeBefore": "2023-11-20T12:15:00Z",
            "pageNumber": "1",
            "collapseOn": 1,
            "eventGroupIncludes": [
                1
            ],
            "collapseLimit": "25"
        }

        # edit eventTimeAfter to be 5 mins before current time in UTC and eventTimeBefore to be 12hrs 5 mins after current time in UTC
        now = datetime.now()
        now = now.strftime(self.format_string)
        now = datetime.strptime(now, self.format_string)
        now = now.timestamp()
        now = now - 300
        now = datetime.fromtimestamp(now)
        now = now.strftime(self.format_string)
        data["eventTimeAfter"] = now
        now = datetime.now()
        now = now.strftime(self.format_string)
        now = datetime.strptime(now, self.format_string)
        now = now.timestamp()
        now = now + 43500
        now = datetime.fromtimestamp(now)
        now = now.strftime(self.format_string)
        data["eventTimeBefore"] = now

        response = self.new_post("https://www.goat.com/api/v1/consumer-search/event-search", json=data, headers=self.headers).json()

        for event in response["resultItems"]:
            for key in event:
                realEvent = event[key]
                eventId = realEvent["id"]
                eventTitle = realEvent["title"]
                try:
                    # if "played" in realEvent or "hasPlayed" in realEvent:
                    #     if realEvent["played"] == True or realEvent["hasPlayed"] == True:
                    #         self.print(f"Skipping {eventTitle}...", "yellow")
                    #         continue

                    self.print(f"Found {eventTitle}! - {eventId}")
                    self.print("Finding Trivia URL...", "blue")

                    response = self.session.get(f"https://www.goat.com/blackfriday?triviaId={eventId}&app=y", headers=self.headers).text

                    triviaUrl = f"https://www.goat.com" + response

                    self.print("Getting Trivia...")
                    response = self.session.get(triviaUrl, headers=self.headers)

                    csrfToken = (response.cookies.get("csrf"))

                    self.print("Starting Trivia Session...")

                    browserHeaders = {
                        'accept': 'application/json',
                        'sec-fetch-site': 'same-origin',
                        'accept-language': 'en-US,en;q=0.9',
                        'sec-fetch-mode': 'cors',
                        'origin': 'https://www.goat.com',
                        'user-agent': 'GOAT/2.66.2 (iPhone; iOS 17.0.3; Scale/3.00) Locale/en',
                        'connection': 'keep-alive',
                        'x-csrf-token': csrfToken,
                        'sec-fetch-dest': 'empty',
                        'content-type': 'text/plain;charset=UTF-8',
                    }

                    self.headers["x-csrf-token"] = csrfToken

                    data = '{"gameId":"' + eventId + '"}'

                    response = self.new_post(
                        'https://www.goat.com/remote-goat-trivia-api/v1/trivia/user-entered-trivia-lobby',
                        headers=browserHeaders,
                        data=data,
                        proxy=reqProxy,
                    )
                    if response.status_code != 200:
                        self.print("Error Starting Trivia Session!", "red")
                        self.print(response.text)
                        exit()


                    data = {"gameId":eventId}
                    response = self.new_post("https://www.goat.com/remote-goat-trivia-api/v1/trivia/start-session-or-get-primary-questions", json=data, headers=browserHeaders).json()
                    if "code" in response:
                        if response["code"] == 3:
                            self.print(response["details"])
                            continue
                    
                    count = 0
                    for question in response["answerablePrimaryQuestionsList"]:
                        count += 1
                        questionId = question["questionId"]
                        questionText = question["questionText"]
                        questionImage = question["questionAsset"]["url"]

                        answerTxt = ""
                        for answer in question["answersList"]:
                            answerTxt += f"{answer['answerId']}:{answer['answerText']}\n"
                        
                        # self.print(f"\n\n\nQuestion: {questionText}")
                        # self.print(f"Answers:\n{answerTxt}")

                        self.print("Getting Answer [AI]", "blue")

                        if questionText in aiAnswers:
                            self.print("Answer Found in Cache - Sleeping 2s...")
                            correctAnswer = aiAnswers[questionText]
                            time.sleep(2)
                        else:
                            
                            response = self.client.chat.completions.create(
                            model="gpt-4-vision-preview",
                            messages=[
                                {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"""You're being quizzed - this image may go with the question or the answer. The possible answers are - 
                                    {answerTxt}
                                    The question is: {questionText}? Respond with the answer ID only, not the answer itself."""},
                                    {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": questionImage,
                                    },
                                    },
                                ],
                                }
                            ],
                            max_tokens=300,
                            )

                            correctAnswer = (response.choices[0].message.content)
                            correctAnswer = correctAnswer.split("\n")[0].strip()
                    

                            aiAnswers[questionText] = correctAnswer


                        self.print(f"Question: {questionText} | Answer: {correctAnswer}", "green")

                        data = {"gameId":eventId,"questionId":questionId,"answerId":correctAnswer}
                        response = self.new_post("https://www.goat.com/remote-goat-trivia-api/v1/trivia/answer-primary-question", json=data, headers=browserHeaders).json()
                        # print(response)
                        sessioncorrectAnswers = response["session"]["numberOfQuestionsAnswered"]

                        self.print(f"Correct Answers: {sessioncorrectAnswers}", "green")

                    self.print("Getting Game Results...", "blue")

                    data = {"gameId":eventId}
                    response = self.new_post("https://www.goat.com/api/v1/trivia/get-session-results", json=data, headers=self.headers)

                    response = response.json()

                    correctAnswers = response["accuracy"]["totalCorrect"] / 5 * 100
                    speed = response["speed"]
                    finishTime = response["finishTime"]
                    ticketsEarned = response["ticketsEarned"]

                    self.print(f"Correct Answers: {correctAnswers}% | Speed: {speed} | Finish Time: {finishTime} | Tickets Earned: {ticketsEarned}", "green")

                except Exception as e:
                        print(e)
                        self.print("Error getting Trivia - " + eventId + " - " + eventTitle, "red")
                        continue
                
                    

        self.print("Getting Latest Tix...", "blue")

        data = {}
        response = self.new_post("https://www.goat.com/api/v1/social_game/get-latest-eligible-game", json=data, headers=self.headers).json()
        if "gameId" not in response:
            self.print("No Latest Tix Found!", "red")
        else:
            latestGameId = response["gameId"]

            data = {"gameId":latestGameId}
            self.print("Starting Latest Tix...", "blue")

            response = self.new_post("https://www.goat.com/api/v1/social_game/play-game", json=data, headers=self.headers)
            if response.status_code != 200:
                self.print("Latest Tix Already Claimed!", "yellow")
            else:
                response = response.json()
                prizeName = response["prize"]["prizeName"]
                self.print(f"Claimed Latest Tix! Prize: {prizeName}", "green")
            
                self.print("Sharing Ticket Post - SOCIALS", "blue")

                for x in range(10):
                    data = {
                            "id":latestGameId,
                            "type":1,
                            "itemType":9,
                            "channelType":x
                    }
                    response = self.new_post("https://www.goat.com/api/v1/community_sharing/share", json=data, headers=self.headers)
                    if response.status_code == 200:
                        self.print(f"Shared on channel {x}!", "green")
            
            
            
        
        self.print('Getting Ticket Count...', "blue")
        data = {}

        response = self.new_post("https://www.goat.com/api/v1/achievement_tickets/get-count", json=data, headers=self.headers).json()
        ticketCount = response["count"]

        

        self.print(f"Ticket Count: {ticketCount}", "green")

        # save to account data
        with open("accounts.json","r") as accounts:
            currentData = json.load(accounts)
            currentData["accounts"][index]["ticketCount"] = ticketCount
        modify_account(index, currentData["accounts"][index])
    def enterDrop(self):
        # create needed variables
        email = self.account["email"]
        password = self.account["password"]
        username = self.account["username"]
        userId = self.account["userId"]
        authToken = self.account["authToken"]
        deviceHash = self.account["deviceHash"]
        deviceId = self.account["deviceId"]
        reqProxy = self.account["proxy"]
        index = self.account["index"]
        userFirstName = self.account["userFirstName"]
        userLastName = self.account["userLastName"]
        addressId = self.account["addressId"]
        billingId = self.account["billingId"]

        self.print("Fetching Drop Info...", "blue")
        
        
        data = {"dropIds":[self.dropId]}

        response = self.new_post("https://www.goat.com/api/v1/drops/get-user-drops", json=data, headers=self.headers).json()
        drop = response["drops"][0]
        dropName = drop["name"]

        dropId = drop["id"]
        dropStartTime = drop["startTime"]
        dropImage = drop["productAssets"][0]["imageUrl"]

        totalPriceCents = drop["priceCents"]
        self.print(f"Total Price: ${int(totalPriceCents)/100}", "green")

        # self.print(drop)
        if "entryTickets" in drop:
            dropTicketCost = drop["entryTickets"][0]["ticketCount"]
            self.print(f"Drop Ticket Cost: {dropTicketCost}", "green")

        self.print(f"Found Drop: {dropName}")

        if drop["ticketsUnlocked"] == False:
            self.print("Unlocking Drop...", "yellow")
            data = {"dropId":dropId}

            response = self.new_post("https://www.goat.com/api/v1/drops/unlock-drop-tickets", json=data, headers=self.headers)
            if response.status_code == 400:
                if response.headers.get("Grpc-Message") == "drop already unlocked with tickets":
                    self.print("Drop Already Unlocked!", "green")
                else:
                    self.print(f"ERROR - " + response.headers.get("Grpc-Message"), "red")
                    exit()
            response = response.json()

        newStartTime = (datetime.strptime(dropStartTime, "%Y-%m-%dT%H:%M:%SZ"))
        # print(datetime.utcnow())

        # wait until drop starts
        now = datetime.now().timestamp()
        dropStart = datetime.strptime(dropStartTime, "%Y-%m-%dT%H:%M:%SZ").timestamp()

        # adjust to PST from UTC
        dropStart -= 28800


        if True:
            random3Letters = ''.join(random.choice(string.ascii_uppercase) for i in range(3)).upper()
            randomApt = str(random.randint(1,125))

            self.print("Adding Address...", "blue")
            data = {
                'address[phone]': f"925{random.randint(1000000,9999999)}",
                'address[name]': userFirstName + " " + userLastName,
                'address[city]': ' ',
                'address[postalCode]': ' ',
                'address[id]': '-1',
                'address[address1]':  '  ' + random3Letters,
                'address[address2]': 'Apt ' + randomApt,
                'address[addressType]': 'shipping',
                'address[state]': 'CA',
                'address[country]': 'United States',
                'address[countryCode]': 'US',
            }
            response = self.new_post('https://www.goat.com/api/v1/addresses', headers=self.headers, data=data)
            if "id" not in response.json():
                self.print("Error Adding Address!", "red")
                self.print(response.json())
                exit()
            addressId = response.json()["id"]

            

            self.print("Adding Billing Address...", "blue")
            data = {
                'address[phone]': f"925{random.randint(1000000,9999999)}",
                'address[name]': userFirstName + " " + userLastName,
                'address[city]': ' ',
                'address[postalCode]': ' ',
                'address[id]': '-1',
                'address[address1]': '  ' + random3Letters,
                'address[address2]': 'Apt ' + randomApt,
                'address[addressType]': 'billing',
                'address[state]': 'CA',
                'address[country]': 'United States',
                'address[countryCode]': 'US',
            }
            response = self.new_post('https://www.goat.com/api/v1/addresses', headers=self.headers, data=data)
            addressIdBilling = response.json()["id"]

            self.print("Adding Billing...")
            self.print("Getting Stripe Session...", "blue")

            json_data = {
                'tag': '22.8.1',
                'src': 'ios-sdk',
                'a': {
                    'c': {
                        'v': 'en_US',
                    },
                    'd': {
                        'v': 'iPhone14,5 17.0.3',
                    },
                    'f': {
                        'v': '390w_844h_3r',
                    },
                    'g': {
                        'v': '-8',
                    },
                },
                'v2': 1,
                'b': {
                    'd': '483a0936-d71f-4e2a-b7eb-9738572a4967f54359',
                    'm': True,
                    'k': 'GOAT',
                    'e': '',
                    'o': '17.0.3',
                    'l': '2.66.2',
                    's': 'iPhone14,5',
                },
            }

            response = requests.post('https://m.stripe.com/6', headers={
                'host': 'm.stripe.com',
                'user-agent': 'GOAT/1 CFNetwork/1474 Darwin/23.0.0',
                'connection': 'keep-alive',
                'accept': '*/*',
                'accept-language': 'en-US,en;q=0.9',
                'content-type': 'application/json',
                # 'cookie': 'm=e91fa4bb-0418-47b5-be1b-8c51a90aa03e7e2bfa',
            }, json=json_data).json()
            muid = response["muid"]
            guid = response["guid"]

            self.print("Getting Stripe Token...", "blue")

            data = {
                'card[address_city]': ' ',
                'card[address_country]': 'US',
                'card[address_line1]': '  ' + random3Letters,
                'card[address_line2]': '',
                'card[address_zip]': ' ',
                'card[cvc]': self.cardCvv,
                'card[exp_month]': self.cardExpMonth,
                'card[exp_year]': self.cardExpYear,
                'card[name]': userFirstName + ' ' + userLastName,
                'card[number]': self.cardNumber.replace(" ", ""),
                'guid': guid,
                'muid': muid,
                'payment_user_agent': 'stripe-ios/22.8.1; variant.legacy'
            }

            response = requests.post('https://api.stripe.com/v1/tokens', headers={
                'host': 'api.stripe.com',
                'accept': '*/*',
                'authorization': 'Bearer pk_live_eVTnJ0YFSiOvBUVnyhbC0Jfg',
                'accept-language': 'en-US,en;q=0.9',
                'x-emb-st': '1700195610653',
                'stripe-version': '2020-08-27',
                'user-agent': 'GOAT/1 CFNetwork/1474 Darwin/23.0.0',
                'x-stripe-user-agent': '{"type":"iPhone14,5","bindings_version":"22.8.1","os_version":"17.0.3","model":"iPhone","vendor_identifier":"79CC76EF-693C-4C92-80EF-2D50EAA6EAEA","lang":"objective-c"}',
                'connection': 'keep-alive',
                'x-emb-id': '8A12E4BE865C4494A6799BBF3A121173',
                'content-type': 'application/x-www-form-urlencoded',
            }, data=data).json()

            stripeToken = response["id"]

            self.print("Adding Billing...", "blue")

            data = {
                'billingInfo[billingAddressId]': str(addressIdBilling),
                'billingInfo[paymentType]': 'card',
                'billingInfo[name]': userFirstName + " " + userLastName,
                'billingInfo[processorName]': 'stripe',
                'billingInfo[stripeToken]': stripeToken,
            }
            response = self.new_post('https://www.goat.com/api/v1/billing_infos', headers=self.headers, data=data).json()
            if "id" not in response:
                self.print("Error Adding Billing!", "red")
                self.print(response)
                exit()
            billingId = response["id"]
            # save to account data
            with open("accounts.json","r") as accounts:
                currentData = json.load(accounts)
                currentData["accounts"][index]["billingId"] = billingId
                currentData["accounts"][index]["addressId"] = addressId
            modify_account(index, currentData["accounts"][index])
        # else:
        #     addressId = self.account["addressId"]
        #     billingId = self.account["billingId"]
        
        self.print(f"Waiting for drop to start - {newStartTime}", "blue")

        while datetime.strptime(dropStartTime, "%Y-%m-%dT%H:%M:%SZ") > datetime.utcnow():
            time.sleep(0.1)

        # while timeToDrop > 0:
        #     self.print(f"Waiting for drop to start - {round(timeToDrop)}s", "yellow")
        #     if timeToDrop > 0:
        #         time.sleep(1)
        #     timeToDrop -= 1

        # self.print(f"\033[95mWaiting for drop to start - {round(timeToDrop)}s\033[0m...")
        # if timeToDrop > 0:
        #     time.sleep(timeToDrop)
        
        self.print("Getting Drop Info...", "blue")
        
        error = True
        while error:
            try:
                response = self.new_post("https://www.goat.com/api/v1/drops/get-user-drops", json={"dropIds":[self.dropId]}, headers=self.headers).json()
                error = False
                break
            except:
                self.print("Error getting Drop Info!", "red")
                time.sleep(1)
        while "drops" not in response:
            self.print(response)
            time.sleep(1)
            self.print("Retrying to get drop...", "yellow")
            response = self.new_post("https://www.goat.com/api/v1/drops/get-user-drops", json=data, headers=self.headers).json()
        drop = response["drops"][0]
        sizes = drop["sizeStockStatuses"]
        # create new size array that's in a random order
        sizes = random.sample(sizes, len(sizes))
        
        chosenSize = None
        
        for size in sizes:
            if "inStock" in size:
                # make sure only chooses size 10.0 and above
                if size["inStock"] == True and float(size["size"]) >= 0.0:
                    chosenSize = size["size"]
                
        while(chosenSize == None):
            self.print("No sizes in stock! Retrying...", "yellow")
            time.sleep(0.5)
            response = self.new_post("https://www.goat.com/api/v1/drops/get-user-drops", json=data, headers=self.headers).json()
            if "drops" not in response:
                continue
            drop = response["drops"][0]
            sizes = drop["sizeStockStatuses"]
            # shuffle sizes
            sizes = random.sample(sizes, len(sizes))
            

            for size in sizes:
                if size["inStock"] == True and float(size["size"]) >= 0.0:
                    if size["inStock"] == True:
                        chosenSize = size["size"]

        if "captchaAssets" in drop:
            number = random.randint(1,1)
            if number == 0:
                captchaCaption = drop["captchaCaption"]
                self.print("Getting Captcha [API] - " + captchaCaption, "blue")

                content = [
                    {"type": "text", "text": f"Which image most likely represents the following product: {dropName}? Respond with the index ONLY and nothing else."},
                ]
                for captcha in drop["captchaAssets"]:
                    content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": captcha["imageUrl"],
                                        "detail":"low"
                                    },
                                })
                    self.print(captcha["imageUrl"])
                    

                response = self.client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                )

                captchaAnswer = (response.choices[0].message.content)

                captchaAnswer = int(captchaAnswer.split("\n")[0].strip())

                captchaId = drop["captchaAssets"][captchaAnswer-1]["id"]
                captchaImage = drop["captchaAssets"][captchaAnswer-1]["imageUrl"]

                self.print(f"Likely Correct Captcha Data: {captchaImage},{captchaId},{captchaAnswer}", "green")


                self.print("Verifying CAPTCHA...", "blue") 

                data = {"dropId":dropId,"captchaAssetId":captchaId}
                response = self.new_post("https://www.goat.com/api/v1/drops/submit-drop-captcha", json=data, headers=self.headers)
                # check if valid json
                try:
                    response = response.json()
                except:
                    self.print("Error Verifying CAPTCHA - JSON ERROR!", "red")
                    self.print(response.text)
                    self.print(response.status_code)
                    self.print(response.headers)
                    exit()
            else:
                captchaAnswer = 1
                captchaId = drop["captchaAssets"][captchaAnswer-1]["id"]
                captchaImage = drop["captchaAssets"][captchaAnswer-1]["imageUrl"]

                self.print(f"Likely Correct Captcha Data [Guessed]: {captchaImage},{captchaId},{captchaAnswer}", "green")
                self.print("Verifying CAPTCHA...", "blue")

                data = {"dropId":dropId,"captchaAssetId":captchaId}
                response = self.new_post("https://www.goat.com/api/v1/drops/submit-drop-captcha", json=data, headers=self.headers)
                
                if "captchaAssets" in response.json():
                    self.print("Guessed CAPTCHA Wrong, Defaulting to API...", "red")

                    captchaCaption = drop["captchaCaption"]
                
                    self.print("Getting Captcha [API] - " + captchaCaption, "blue")

                    content = [
                        {"type": "text", "text": f"Which image most likely represents the following product: {dropName}? Respond with the index ONLY and nothing else."},
                    ]
                    for captcha in drop["captchaAssets"]:
                        content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": captcha["imageUrl"],
                                            "detail":"low"
                                        },
                                    })
                        self.print(captcha["imageUrl"])
                        

                    response = self.client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": content,
                            }
                        ],
                    )

                    captchaAnswer = (response.choices[0].message.content)

                    captchaAnswer = int(captchaAnswer.split("\n")[0].strip())

                    captchaId = drop["captchaAssets"][captchaAnswer-1]["id"]
                    captchaImage = drop["captchaAssets"][captchaAnswer-1]["imageUrl"]

                    self.print(f"Likely Correct Captcha Data: {captchaImage},{captchaId},{captchaAnswer}", "green")


                    self.print("Verifying CAPTCHA...", "blue") 

                    data = {"dropId":dropId,"captchaAssetId":captchaId}
                    response = self.new_post("https://www.goat.com/api/v1/drops/submit-drop-captcha", json=data, headers=self.headers)
                    # check if valid json
                    try:
                        response = response.json()
                    except:
                        self.print("Error Verifying CAPTCHA - JSON ERROR!", "red")
                        self.print(response.text)
                        self.print(response.status_code)
                        self.print(response.headers)
                        exit()
                else:
                    self.print("Guessed CAPTCHA Correct!", "green")
                    
            
            response = response.json()["data"]

            if "captchaAssets" not in response:
                self.print("CAPTCHA Verified!", "green")
            else:
                self.print("Error Verifying CAPTCHA!", "red")
                exit()
            dropSlug = response["productTemplateSlug"]
        else:
            dropSlug = drop["productTemplateSlug"]
            

        self.print("Entering Drop - " + dropName + " - " + chosenSize + "...", "blue")

        data = {
            "productTemplateSlug":dropSlug,
            "size":chosenSize,
            "addressId":str(addressId),
            "billingInfoId":str(billingId)
        }

        response = self.new_post("https://www.goat.com/api/v1/order-reservation/build-reservation", json=data, headers=self.headers)
        if response.status_code != 200:
            self.print("Error entering Drop!", "red")
            self.print(response.text)
            self.print(response.status_code)
            self.print(response.headers)
            exit()
        response = response.json()
        reservationId = response["reservationId"]
        totalPriceCents = response["finalPrice"]["cents"]

        self.headers["x-emb-st"] = str(round(datetime.timestamp(datetime.now()) * 1000))
        self.headers["x-emb-id"] = ''.join(random.choices(string.digits + 'ABCDEF', k=32))

        self.print(f"Submitting Drop - Total: ${int(totalPriceCents)/100}...", "blue")

        data = {"reservationId":reservationId}

        response = self.new_post("https://www.goat.com/api/v1/order-reservation/submit-reservation", json=data, headers=self.headers).json()

        # self.print(response)

        while True:
            self.print("Checking Drop Status...", "blue")
            try:
                response = self.new_post("https://www.goat.com/api/v1/order-reservation/get-reservation-status", json=data, headers=self.headers).json()
                
                # self.print(response)
                if response["status"] == "ORDER_FAILED":
                    self.print("Drop Entry Not Selected!", "red")
                    self.sendWebhook("Drop Entry Not Selected!", response["status"], "FF0000", dropImage, str(int(totalPriceCents)/100), f"||{self.username}||", chosenSize, dropName)
                    exit()
                elif response["status"] == "ENQUEUED":
                    self.print("Drop Entry Queued!", "yellow")
                    pass
                    # exit()
                elif response["status"] == "ORDER_CONFIRMED":
                    self.print("Drop Entry Selected!", "green")
                    self.sendWebhook("Drop Entry Selected!", response["status"], "E0B0FF", dropImage, str(int(totalPriceCents)/100), f"||{self.username}||", chosenSize, dropName)
                    exit()
                    
                else:
                    self.print("Drop Entry Selected (POTENTIAL)!", "green")
                    self.print(response)
            except Exception as e:
                self.print("Error Checking Drop! Retrying...", "red")
            time.sleep(5)
    def createAccount(self):
        proxyToUse = random.choice(self.proxyList)

        self.headers["authorization"] = f"Token token=\"\""

        self.print(f"Getting session for NEW ACCOUNT...", "yellow")

        # chosen = random.choice(self.accountData["accounts"])

        cookieData = self.genPx()
        cookie = cookieData["pxCookie"]
        reqProxy = {
            "http": cookieData["proxy"],
            "https": cookieData["proxy"]
        }
        # get time in ms
        now = datetime.now()
        timestamp = round(datetime.timestamp(now) * 1000)

        self.headers["x-emb-st"] = str(timestamp)
        self.headers["x-px-authorization"] = f"3:{cookie}"
        # self.headers["authorization"] = f"Token token=\"{self.authToken}\""

        devideId = str(uuid4())
        deviceHash = hashlib.sha256(devideId.encode()).hexdigest()

        self.print("Creating Account | Device ID: " + devideId + " | Device Hash: " + deviceHash)

        fake = faker.Faker()
        fakeName = fake.name()
        fakePassword = fake.password( length=16, special_chars=True, digits=True, upper_case=True, lower_case=True)

        email = fakeName.replace(" ","").lower() + str(random.randint(1000000,500000000000)) + "@gmail.com"
        data = {
            'emailRegistration': '1',
            'user[name]': fakeName,
            'user[region]': 'US',
            'user[email]': email,
            'user[password]': fakePassword,
        }

        response = self.new_post('https://www.goat.com/api/v1/users', headers=self.headers, data=data)

        data = response.json()

        userId = data["id"]
        email = data["email"]
        username = data["username"]
        authToken = data["authToken"]

        self.print("Setting defualt region & currency...")
        self.headers["authorization"] = f"Token token=\"{authToken}\""

        data = {
            'region_preferences[country]': 'US',
        }

        response = self.new_post(f'https://www.goat.com/api/v1/region_preferences', headers=self.headers, data=data)

        data = {
            'currency': 'USD',
        }

        response = self.session.put(
            'https://www.goat.com/api/v1/currencies/set_preferred_currency',
            headers=self.headers,
            data=data,
            proxy=reqProxy
        )

        self.print("Account Preferences Added!")
        self.print("Binding Device...")

        data = {
            'device[devicePlatform]': 'iOS',
            'device[deviceOs]': 'iOS 17.0.3',
            'device[currentIp]': reqProxy["http"].split("@")[1].split(":")[0],
            'device[sandbox]': '0',
            'device[deviceHardware]': 'iPhone14,5',
            'device[deviceToken]': deviceHash,
            'device[deviceIdentifier]': devideId.upper(),
            'device[advertisingId]': '00000000-0000-0000-0000-000000000000',
        }

        response = self.new_post('https://www.goat.com/api/v1/devices', headers=self.headers, data=data).json()

        if response["success"]:
            print("Device Bound!")
        else:
            print("Error Binding Device!")
            print(response)

        self.print("Adding to database...")


        with lock:
            with open("accounts.json", "r") as file:
                accounts = json.load(file)

            # Modify the account
            accounts["accounts"].append({
                "email": email,
                "username": username,
                "password": fakePassword,
                "authToken": authToken,
                "userId": userId,
                "deviceHash": deviceHash,
                "deviceId": devideId,
                "userFirstName": fakeName.split(" ")[0],
                "userLastName": fakeName.split(" ")[1],
                "addressId":"0",
                "billingId":"0",
                "proxy": reqProxy["http"],
                "pxCookie": cookie
            })

            # Write the changes back to the file
            with open("accounts.json", "w") as file:
                json.dump(accounts, file, indent=4)

        self.print(f"Created account with email: {email} and username: {username} and password: {fakePassword}", "green")

        self.account["userFirstName"] = fakeName.split(" ")[0]
        self.account["userLastName"] = fakeName.split(" ")[1]
        self.account["addressId"] = "0"
        self.account["billingId"] = "0"



