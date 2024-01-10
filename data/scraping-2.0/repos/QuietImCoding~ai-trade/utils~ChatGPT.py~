import openai
import os

class chatGPT: 

    openai.api_key = os.environ['OPENAPIKEY']
    
    def update_market(self, market):
        market_update = """DAY 1:
        INVENTORY:
            - """ + str(self.coins) + " coins\n" + '\n'.join([ f"\t- {self.inventory[k]} {k}" for k in self.inventory ]) + """
        FRUIT AVAILABLE:\n""" + '\n'.join([f"\t- {item} : {market[item]} coins" for item in market])
        print(market_update)
        self.messages.append({"role": "system", "content": market_update})

    def parse_commands(self, cmdstring, market):
        cmdlist = cmdstring.split('\n')
        for c in cmdlist:
            fields = c.split()
            fields[1] = int(fields[1])
            pnl = (-1 if fields[0] == "BUY" else 1) * fields[1] * market[fields[2]]
            self.coins = self.coins + pnl
            if fields[2] in self.inventory and fields[0] == "SELL":
                self.inventory[fields[2]] -= fields[1]
            elif fields[2] not in self.inventory and fields[0] == "SELL":
                exit("Tried to sell asset we don't have")
            elif fields[2] not in self.inventory and fields[0] == "BUY":
                self.inventory[fields[2]] = fields[1]
            else:
                self.inventory[fieds[2]] += fields[1]

    def query(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
        return completion['choices'][0]['message']['content']

    def __init__(self):
        self.day = 1
        self.coins = 1000
        self.inventory = {}
        self.messages = [
            {"role": "system", "content": "You are playing a turn-based video game where you are a fruit salesman. You have an inventory consisting of coins and fruit and want to get as many coins as possible buying and selling fruits."},
            {"role": "system", "content": "Every turn you will be provided with a list of your inventory and the market price of each fruit you own. On each turn you can use any combination of the commands 'BUY <number> <FRUIT>' and 'SELL <number> <FRUIT>'. You cannot buy and sell the same fruit multiple times in a single turn. Do not respond with anything but a newline separated list of game commands"}
        ]

    
if __name__ == '__main__':
    gpt = chatGPT()
    market = { "apples" : 10, "blueberries": 2, "bananas": 4, "raspberries": 6}
    gpt.update_market(market)
    gptsays = gpt.query()
    print(gptsays)
    gpt.parse_commands(gptsays, market)
    gpt.update_market(market)
    gptsays = gpt.query()
    print(gptsays)


