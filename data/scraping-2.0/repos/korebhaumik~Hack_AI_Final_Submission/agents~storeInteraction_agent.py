from uagents import Agent, Bureau, Context, Model
from uagents.network import wait_for_tx_to_complete
from uagents.setup import fund_agent_if_low
from openai import OpenAI
import json
import os
from helpers.redisClient import redis_client
from helpers.dbConnect import bookCollection, purchasesCollection
from controllers.newPurchase import newPurchase
from controllers.getBookPrice import get_book_price
import datetime

from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

DENOM = "atestfet"

OpenAi_client = OpenAI(api_key=os.environ.get("OPENAIKEY"))


def read_query(filename):
    try:
        dataRaw = redis_client.get(filename)
        if dataRaw and dataRaw != b"":
            data = json.loads(dataRaw)
            return data["isActive"], data["prompt"], data["isReceived"]
    except Exception as e:
        print(e)


async def reset_file(filename, output):
    try:
        data = {
            "isActive": False,
            "status": 0,
            "prompt": "",
            "isReceived": False,
            "isDone": True,
            "output": output,
        }
        redis_client.set(filename, json.dumps(data))
    except Exception as e:
        print(e)


def update_receive(filename, state):
    try:
        data = {}
        dataRaw = redis_client.get(filename)
        data = json.loads(dataRaw)
        data["isReceived"] = state
        redis_client.set(filename, json.dumps(data))
    except Exception as e:
        print(e)


main_agent_store = Agent(name="main_agent_store", seed="main_agent_store")

fetch_all_agent = Agent(name="fetch_all_agent", seed="fetch_all_agent")

fetch_doc_agent = Agent(name="fetch_doc_agent", seed="fetch_doc_agent")

compare_doc_agent = Agent(name="compare_doc_agent", seed="compare_doc_agent")

purchase_info_agent = Agent(name="purchase_info_agent", seed="purchase_info_agent")

user = Agent(name="user", seed="user secret phrase")

merchant = Agent(name="merchant", seed="merchant secret phrase")

fund_agent_if_low(user.wallet.address())
fund_agent_if_low(merchant.wallet.address())


class Message(Model):
    message: str


class PaymentRequest(Model):
    wallet_address: str
    denom: str
    amount: int
    pricePerBook: int
    quantity: int
    title: str
    address: str


class TransactionInfo(Model):
    tx_hash: str
    wallet_address: str
    amount: int
    pricePerBook: int
    quantity: int
    title: str
    address: str


@main_agent_store.on_interval(period=1)
async def process_query(ctx: Context):
    try:
        is_active, query, isReceived = read_query("store")
        if is_active and not isReceived:
            ctx.logger.info(f"Received query: {query}")
            update_receive("store", state=True)

            ## Redirect Logic
            statusRaw = await gpt_redirect(query)

            status = json.loads(statusRaw)

            if status["type"] == "1":
                # all books info
                await ctx.send(fetch_all_agent.address, Message(message=query))
            if status["type"] == "2":
                # one book info
                await ctx.send(
                    fetch_doc_agent.address, Message(message=status["title"])
                )
            if status["type"] == "3":
                # do purchase
                # output = "Book Name: " + status["title"] +  ", calling 'purchase' agent..."
                bookTitle = status["title"]
                purchaseQuantity = status["quantity"]
                if bookTitle:
                    pricePerBook = get_book_price(bookTitle)
                if bookTitle and purchaseQuantity and pricePerBook is not None:
                    purchase_details = {
                        "title": bookTitle,
                        "pricePerBook": pricePerBook,
                        "quantity": purchaseQuantity,
                        "user_walet_address": "fetch146rgrqzxgnwquk6pchxdzrm5al7wxdjwxg6ym7",
                        "address": "Marol, Andheri E, Mumbai, 4000059",
                    }
                    print(purchaseQuantity)
                    print(pricePerBook)
                    amount = int(purchaseQuantity) * int(pricePerBook)
                    ctx.logger.info(
                        f"\033[96m TRANSACTION OF {amount} {DENOM} REQUESTED"
                    )
                    await ctx.send(
                        user.address,
                        PaymentRequest(
                            wallet_address=str(purchase_details["user_walet_address"]),
                            denom=DENOM,
                            amount=amount,
                            pricePerBook=purchase_details["pricePerBook"],
                            quantity=purchase_details["quantity"],
                            title=purchase_details["title"],
                            address=purchase_details["address"],
                        ),
                    )
            if status["type"] == "4":
                # purchase history
                await ctx.send(
                    purchase_info_agent.address, Message(message="Just do it bro")
                )
                # await reset_file("store", output=output)
            if status["type"] == "5":
                output = status["msg"]
                await reset_file("store", output=output)
    except Exception as e:
        print(e)


@main_agent_store.on_message(model=Message)
async def main_agent_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"Received message from {sender}: {msg.message}")
        output = msg.message
        await reset_file("store", output=output)
    except Exception as e:
        print(e)


async def gpt_redirect(query):
    try:
        prompt = """
                Your output should be strictly based as mentioned below.
                It is based on the type and only on the type of the query given.

                1) If the given query mentions to fetch data or get all data from the store return {"type":"1"}.

                2) If the given query mentions to fetch to a specific book from the store return {"type":"2", "title": "Title of the Book"}.

                3) If the given query mentions to make a purchase for a particular book return {"type":"3","quantity":"quantity of items","title": "Title of the Book"}.

                4) If the given query mentions to fetch purchase information or previous purchase/orders history return {"type":"4"}.

                5) IF ANY OTHER QUERYIES ARE ASKED OTHER THAN THESE FOUR REPLY WITH {"type":"5", "msg": "Out of my power..."}

                Query should be a case of '1', '2', '3', '4' or '5' from the above conditions.

                For your reference. Available Books are.
                1984, To Kill a Mockingbird, Pride and Prejudice, The Great Gatsby, Moby-Dick, The Shadow of the Wind, The Alchemist, Beloved, Life of Pi, A Thousand Splendid Suns, Slaughterhouse-Five, The Handmaid's Tale, Wuthering Heights, Never Let Me Go, American Gods.
                """

        completion = OpenAi_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{query}"},
            ],
        )
        response = completion.choices[0].message.content
        print(response)
        return response
    except Exception as e:
        print(e)


##############################################################################################################


@fetch_all_agent.on_message(model=Message)
async def main_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"Received message from {sender}")
        docs_string = fetch_docs()
        response = format_docs(docs_string)
        print("Abra", response)
        await ctx.send(main_agent_store.address, Message(message=f"{response}"))
    except Exception as e:
        print(e)


def fetch_docs():
    try:
        print("Fetch docs running")
        cursor = bookCollection.find(
            {}, {"_id": 0, "title": 1, "price": 1, "quantity": 1, "rating": 1}
        )
        docs = list(cursor)
        print("docs are", docs)
        json_string = json.dumps(docs)
        return json_string
    except Exception as e:
        print(e)


def format_docs(query):
    try:
        print("Format docs running")
        prompt = """ The Following are the given docs in json:
            The output should be exactly as formated as follows:
            First line should be the column names: "Title | Price | Rating | Quantity"
            1) title of the book | price | rating | quantity
            2) title of the book | price | rating | quantity
            3) title of the book | price | rating | quantity
            ... and so on. 
            try to seperate it usign space and format it to be even throughout
            """

        completion = OpenAi_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{query}"},
            ],
        )
        return str(f"```{completion.choices[0].message.content}```")
    except Exception as e:
        print(e)


##############################################################################################################


@fetch_doc_agent.on_message(model=Message)
async def main_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"Received message from {sender}")
        bookTitle = msg.message
        docs_string = fetch_doc(bookTitle)
        response = format_doc(docs_string)
        print("Abra", response)
        await ctx.send(main_agent_store.address, Message(message=f"{response}"))
    except Exception as e:
        print(e)


def fetch_doc(name):
    try:
        print("Fetch doc running")
        global bookCollection
        doc = bookCollection.find_one(
            {"title": name},
            {
                "_id": 0,
                "summary": 1,
                "title": 1,
                "price": 1,
                "description": 1,
                "tags": 1,
                "quantity": 1,
            },
        )
        print("docs are", doc)
        json_string = json.dumps(doc)
        return json_string
    except Exception as e:
        print(e)


def format_doc(query):
    try:
        print("Format doc running")
        prompt = """ The Following are the given docs in json:
            The output should be exactly as formated as follows:

            --Title--
            --Price--
            --Description--
            --Tags--
            --Summary--
            --Quantity--
            try to seperate it usign space and format it to be even throughout
            """

        completion = OpenAi_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{query}"},
            ],
        )
        return str(f"```{completion.choices[0].message.content}```")
    except Exception as e:
        print(e)


##############################################################################################################


@compare_doc_agent.on_message(model=Message)
async def main_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"Received message from {sender}")
        docs_string = compare_doc(msg.message)
        response = format_comparison(docs_string)
        print("Abra", response)
        await ctx.send(main_agent_store.address, Message(message=f"{response}"))
    except Exception as e:
        print(e)


def compare_doc(name):
    try:
        print(name)
        global bookCollection
        cursor = bookCollection.find(
            {"price": {"$gte": 20}}, {"_id": 0, "title": 1, "price": 1}
        )
        docs = [doc for doc in cursor]
        print("docs are", docs)
        json_string = json.dumps(docs)
        return json_string
    except Exception as e:
        print(e)


def format_comparison(query):
    try:
        prompt = """
                The Following are the given docs in json:
                The output should be exactly as formated as follows:
                First line should be the column names: "Title | Price | Rating | Quantity"
                1) title of the book | price | rating | quantity
                2) title of the book | price | rating | quantity
                3) title of the book | price | rating | quantity
                ... and so on. 
                try to seperate it usign space and format it to be even throughout
                """

        completion = OpenAi_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{query}"},
            ],
        )
        return str(f"```{completion.choices[0].message.content}```")
    except Exception as e:
        print(e)


##############################################################################################################


@purchase_info_agent.on_message(model=Message)
async def main_message_handler(ctx: Context, sender: str, msg: Message):
    try:
        ctx.logger.info(f"Received message from {sender}")
        docs_string = fetch_purchase_info()
        response = format_purchase_info(docs_string)
        print("Abra", response)
        await ctx.send(main_agent_store.address, Message(message=f"{response}"))
    except Exception as e:
        print(e)


def fetch_purchase_info():
    try:
        cursor = purchasesCollection.find(
            {},
            {
                "_id": 0,
                "title": 1,
                "pricePerBook": 1,
                "quantity": 1,
                "walet_address": 1,
                "datetime": 1,
                "delivery_date": 1,
            },
        )
        docs = [doc for doc in cursor]
        print("docs are", docs)
        json_string = json.dumps(docs)
        return json_string
    except Exception as e:
        print(e)


def format_purchase_info(query):
    try:
        prompt = f""" 
                The Following are the given docs in json:
                The output should be exactly as formated as follows:
                First line should be the column names: "Title | Price | Quantity | Wallet Address | Date | Delivery Date"
                for every purchase the wallet address will be fetch146rgrqzxgnwquk6pchxdzrm5al7wxdjwxg6ym7 and the merchant address will be {merchant.address}
                Followed by :
                1) title of the book | price | quantity | wallet address | merchant address | date | delivery date | address
                2) title of the book | price | quantity | wallet address | merchant address | date | delivery date | address
                3) title of the book | price | quantity | wallet address | merchant address | date | delivery date | address
                ... and so on. 
                try to seperate it usign space and format it to be even throughout
                """

        completion = OpenAi_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{query}"},
            ],
        )
        return str(f"```{completion.choices[0].message.content}```")
    except Exception as e:
        print(e)


############################################################################################################


@user.on_message(model=PaymentRequest, replies=TransactionInfo)
async def send_payment(ctx: Context, sender: str, msg: PaymentRequest):
    try:
        ctx.logger.info(
            f"\033[96m RECIEVED PAYMENT REQUEST FROM {sender} FOR {msg.amount} {DENOM}"
        )
        transaction = ctx.ledger.send_tokens(
            msg.wallet_address, msg.amount, msg.denom, ctx.wallet
        )
        await ctx.send(
            merchant.address,
            TransactionInfo(
                tx_hash=transaction.tx_hash,
                wallet_address=msg.wallet_address,
                amount=msg.quantity * msg.pricePerBook,
                pricePerBook=msg.pricePerBook,
                quantity=msg.quantity,
                title=msg.title,
                address=msg.address,
            ),
        )
    except Exception as e:
        print(e)


@merchant.on_message(model=TransactionInfo)
async def confirm_transaction(ctx: Context, sender: str, msg: TransactionInfo):
    try:
        ctx.logger.info(f"\033[96m PROCESSING TRANSACTION_HASH {msg.tx_hash}")
        tx_resp = await wait_for_tx_to_complete(ledger=ctx.ledger, tx_hash=msg.tx_hash)
        coin_received = tx_resp.events["coin_received"]
        currentDate = datetime.datetime.now()
        if (
            coin_received["receiver"] == str(msg.wallet_address)
            and coin_received["amount"] == f"{msg.amount}{DENOM}"
        ):
            ctx.logger.info(
                f"\033[96m TRANSACTION OF {coin_received} {DENOM} SUCCESSFULL. from {msg.wallet_address} -> merchant"
            )
            foo = newPurchase(
                title=msg.title,
                pricePerBook=msg.pricePerBook,
                quantity=msg.quantity,
                wallet_address=msg.wallet_address,
                datetime=str(currentDate),
                delivery_date=str(currentDate + datetime.timedelta(days=2)),
                delivery_address=msg.address,
            )
            if foo:
                ctx.logger.info(f"\033[96m TRANSACTION RECORDED IN DATABASE")
                prompt = """ The Following is a log from the server of a transaction completed about a book purchase.
                        Format the logs and make it so that it looks like an invoice. Add all the attributes provided and give eqaul spaces to give it a structure.
                        """

                completion = OpenAi_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"{prompt}"},
                        {
                            "role": "user",
                            "content": f"TRANSACTION OF {coin_received} {DENOM} SUCCESSFULL. from wallet adrress {msg.wallet_address}  to merchant wallet address {merchant.address}. title of book bought is {msg.title} , quantity bought is {msg.quantity} time bought is {str(currentDate)} estimated time of delivery address is {msg.address}",
                        },
                    ],
                )
                response = str(f"```{completion.choices[0].message.content}```")
                await ctx.send(main_agent_store.address, Message(message=f"{response}"))
            else:
                ctx.logger.info(f"Failed while putting in Database")
    except Exception as e:
        print(e)
