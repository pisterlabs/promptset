import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import atexit
from middleware import *
import requests


# region chatGPT code
from openai import OpenAI
import os
import json

# Initialize OpenAI client with API key from env var or config.json
client = OpenAI(
    api_key = os.getenv('apitoken') or json.load(open('./secrets/config.json'))['APITOKEN']
)

# Initial message for conversation prompt
STARTCONVOPROMPT = [{"role": "system", "content": "you are an AI chat bot that is helping a user learn more about their crypto wallet and crypto currencies as a whole."}]

# Dictionary to store chat logs for different users
chat_logs = dict()


# Function to call OpenAI GPT-3 API for chat completion
def callAPI(chatId, uinp):
    try:
        if len(uinp) == 0:
            return 'Please provide a message!'
        
        # get the previous data or init
        if chatId not in chat_logs:
            chat_logs[chatId] = STARTCONVOPROMPT.copy()

        chat_logs[chatId].append({"role": "user", "content": "user: " + uinp})

        chat_completion = client.chat.completions.create(
            messages=chat_logs[chatId],
            model="gpt-3.5-turbo",
            stream=True
        )

        retStr = ''
        for part in chat_completion:
            retStr += part.choices[0].delta.content or ""

        chat_logs[chatId].append({"role": "assistant", "content": retStr})

        return retStr #, chat_logs[chatId]
    except Exception as err:
        print(err)
        return str(err)
    
# endregion


# Function to save chat logs to a JSON file
def save_chat_logs():
    with open('./secrets/chat_logs.json', 'w') as file:
        json.dump(chat_logs, file)


# Function to load chat logs from a JSON file
def load_chat_logs():
    global chat_logs
    try:
        with open('./secrets/chat_logs.json', 'r') as file:
            chat_logs = json.load(file)
    except FileNotFoundError:
        pass  # If the file doesn't exist, start with an empty chat_logs dict
    

# Server confs
hostName = "localhost"
serverPort = 8080


# Custom HTTP request handler class
class MyServer(BaseHTTPRequestHandler):

    # Handling endpoints for GET requests
    def do_GET(self):

        # main HTML file
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            f = open('./web/chat.html')
            content = f.read()
            f.close()
            self.wfile.write(bytes(content, "utf-8"))


        # obsolete, but still here for legacy reasons
        elif self.path == "/cryptoScript.js":
            self.send_response(200)
            self.send_header("Content-type", "application/javascript")
            self.end_headers()
            f = open('./web/cryptoScript.js')
            content = f.read()
            f.close()
            
            self.wfile.write(bytes(content, "utf-8"))


        else:
            self.send_response(404)
            self.end_headers()


    # Handling endpoints for POST requests
    def do_POST(self):

        # create a new chat
        if self.path == "/initChat":
            chatId = self.headers.get("chatId")
            if (not chatId): return

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            if chatId not in chat_logs:
                chat_logs[chatId] = STARTCONVOPROMPT.copy()

            self.wfile.write(bytes(json.dumps(chat_logs[chatId]), "utf-8"))


        # Call the ChatGPT API
        elif self.path == "/callAPI":
            content_length = int(self.headers.get("Content-Length"))

            chatId = self.headers.get("chatid")
            if (not chatId): return

            inparr = self.rfile.read(content_length).decode("utf-8")
            if (len(inparr) == 0): return

            retStr = callAPI(chatId, inparr)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            self.wfile.write(bytes(retStr, "utf-8"))


        # Call our AI model to classify spending
        elif self.path == "/isExcess":
            content_length = int(self.headers.get("Content-Length"))

            inparrstr = self.rfile.read(content_length).decode("utf-8")
            if (len(inparrstr) == 0): return

            inparr = []
            try:
                for i in inparrstr.split(','):
                    inparr.append(float(i))
            except Exception:
                return self.send_response(422)  # unprocessable entry
            
            outp = classifyTransaction(inparr)
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            self.wfile.write(bytes(json.dumps(outp), "utf-8"))

        
        # Get the current price of Etherium, as well as converting the USD amount to ETH
        elif self.path == '/getEthPrice':
            try:
                kraken_response = requests.get('https://api.kraken.com/0/public/Ticker?pair=ETHUSD')
                kraken_data = kraken_response.json()
                if (kraken_data['error']): return self.send_response(500)

                ETHtoUSD = float(kraken_data['result']['XETHZUSD']['a'][0])

                content_length = int(self.headers.get("Content-Length"))
                USDamt = self.rfile.read(content_length).decode("utf-8")
                ethAMT = float(USDamt) / ETHtoUSD

                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()

                self.wfile.write(bytes(json.dumps({"ETHAMT": ethAMT, "ETHTOUSD": ETHtoUSD}), "utf-8"))
                
            except Exception as e:
                print('Error:', str(e))
                return json.loads({'error': 'Internal Server Error'}), 500
            
        
        # Get the user's last transaction
        elif self.path == '/getLastTrans':
            content_length = int(self.headers.get("Content-Length"))
            addr = self.rfile.read(content_length).decode("utf-8")
            lastTrans = pyfuncs.getLastTrans(addr)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            self.wfile.write(bytes(lastTrans, "utf-8"))


        # convert MATIC to ETH
        elif self.path == '/MATICtoETH':
            content_length = int(self.headers.get("Content-Length"))
            MATICAMT = self.rfile.read(content_length).decode("utf-8")
            ETHAMT = pyfuncs.MATICtoETH(float(MATICAMT))

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            self.wfile.write(bytes(ETHAMT, "utf-8"))

        # elif self.path == '/getTransAmt':
        #     content_length = int(self.headers.get("Content-Length"))
        #     toAddr = self.rfile.read(content_length).decode("utf-8")
            
        #     ethScanReq = f"https://api.etherscan.io/api?module=account&action=txlist&address={toAddr}&startblock=0&endblock=99999999&page=1&offset=10&sort=asc&apikey=99N6CXU2MTIQZVEF7T6488A3TCRIADFQ2I"
            
        #     print(ethScanReq)
        #     ethTransResp = requests.get(ethScanReq)
        #     # if (ethTransResp['error']): return
        #     print(ethTransResp)


        # Get the user's incoming and outgoing transactions
        elif self.path == '/getTransactions':
            content_length = int(self.headers.get("Content-Length"))
            ADDR = self.rfile.read(content_length).decode("utf-8")
            [inMap, outMap] = pyfuncs.coalateTransactions(ADDR)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()

            self.wfile.write(bytes(json.dumps([inMap, outMap]), "utf-8"))

        else:
            print("UNKNOWN ENDPOINT", self.path)
            self.send_response(404)



# save the user's chat logs to a local JSON file
atexit.register(save_chat_logs)

# Loading existing chat logs from a local JSON file
load_chat_logs()

webServer = HTTPServer((hostName, serverPort), MyServer)
print("Server started http://%s:%s" % (hostName, serverPort))

# Keep the server indefinitely until a KeyboardInterrupt (Ctrl+C)
try:
    webServer.serve_forever()  # BLOCKING CALL
except KeyboardInterrupt:
    pass

webServer.server_close()
print("Server stopped.")
