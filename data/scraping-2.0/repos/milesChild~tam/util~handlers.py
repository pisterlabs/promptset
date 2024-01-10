from telegram.ext import ConversationHandler
from uuid import uuid4
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from util.quote_service import get_quote
import config as cfg
from util import rds
from util.openfigi import get_figi_from_identifier
from datetime import datetime
from typing import Dict, Union, List
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import logging
import openai
import os
import sys
from prompts.client_prompts import CONVERSATION_PROMPT
sys.path.append("..")


logger = logging.getLogger(__name__)

openai.api_key = cfg.openai_key


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )

    return response.choices[0].message["content"]


QUOTES = dict()

RESPONSE_FIELD_MAP = {
    "identifier": "identifier",
    "side": "quote_side",
    "size": "quantity",
    "quote_method": "quote_type",
}

logger = logging.getLogger(__name__)


async def register_cpty(update, context, response):
    pass


async def create_ticket(config):
    """
    Opens a ticket and provides a quote to the client.

    Accepts a JSON of the following format in the "response" field:
    {
        "identifier": "..",
        "side": "..",
        "size": "..",
        "quote_method": ".."
    }
    "identifier" must be a valid ISIN, CUSIP, SEDOL, or recognized ticker symbol.
    "side" must be "BID", "OFFER", or "2WAY" and must reflect the trade action
    being taken by the market maker, i.e. BID means the market maker is buying.
    "size" must be either a quantity of the security or notional value to trade,
    expressed as an integer. Provide -1 for size for a default value.
    "quote_method" must be the reference quote type the client would like to
    execute against. Accepted values are RISK, NAV, TWAP, VWAP, and GMOC. If not
    provided, we assume "RISK" is the quote method.

    Clients are shown one quote which will not update unless the RFQ is modified with the modify handler.    
    """

    telegram_id = config["client_id"]

    cpty_id = rds.run(
        **cfg.hibiscus_db,
        query=f"SELECT counterparty_id FROM hibiscus.telegram_users WHERE telegram_id = '{telegram_id}'",
        return_dict=True
    )[0].get("counterparty_id")

    if not cpty_id:
        return 'User not registered with TAM. Please use /register to be onboarded.'
    elif cpty_id in QUOTES:
        return 'You already have an active quote with us. Please cancel your current quote before requesting new quotes.'
    else:
        identifier = config.get("identifier")
        if not identifier:
            return 'No valid identifier provided to quote on.'
        else:
            figi_dict = get_figi_from_identifier(identifier)
            figi_dict["security"] = figi_dict["name"]
            quote_dict = {
                "event_type": "QUOTE",
                "event_time": datetime.now(),
                "quote_side": config.get("side"),
                "quote_type": config.get('quote_method'),
                "identifier": identifier,
                "counterparty_id": cpty_id,
                "user_id": telegram_id,
                "quantity": config.get("size"),
                "bid_offset": None,
                "bid_quote": None,
                "ask_offset": None,
                "ask_quote": None,
                **figi_dict,
            }
            get_quote(quote_dict)

            send_list = ["event_type", "event_time", "quote_type", "counterparty_id", "user_id",
                         "security", "figi", "quantity", "bid_offset", "bid_quote", "ask_offset", "ask_quote"]
            send_dict = {k: quote_dict[k] for k in send_list}
            rds.insert(**cfg.hibiscus_db, table="transactions", data=send_dict)
            QUOTES[cpty_id] = quote_dict

            if quote_dict['quote_side'] == "2WAY":
                return 'We\'re at {} / {} for size {} of {}'.format(
                    quote_dict['bid_offset'], quote_dict['ask_offset'], quote_dict['quantity'], quote_dict['identifier'])
            elif quote_dict['quote_side'] == "BID":
                return 'We\'re at {} to buy {} of {}'.format(
                    quote_dict['bid_offset'], quote_dict['quantity'], quote_dict['identifier'])
            elif quote_dict['quote_side'] == "OFFER":
                return 'We\'re at {} to sell {} of {}'.format(
                    quote_dict['ask_offset'], quote_dict['quantity'], quote_dict['identifier'])


async def cancel_ticket(config: Dict):
    """
    If the counterparty has an outstanding ticket, cancels the outstanding ticket and
    logs a QUOTE_CANCEL event to the database.

    If the counterparty does not have an active ticket, we do not alter any state or
    log an event, only send a message back.
    """

    telegram_id = config["client_id"]
    cpty_id = rds.run(
        **cfg.hibiscus_db,
        query=f"SELECT counterparty_id FROM hibiscus.telegram_users "
              f"WHERE telegram_id = '{telegram_id}'",
        return_dict=True
    )[0].get("counterparty_id")

    if not cpty_id:
        return 'User not registered with TAM. Please use /register to be onboarded.'

    cancel_dict = QUOTES.pop(cpty_id)
    if cancel_dict:
        send_list = ["event_type", "event_time", "quote_type", "counterparty_id", "user_id",
                     "security", "figi", "quantity", "bid_offset", "bid_quote", "ask_offset", "ask_quote"]
        send_dict = {k: cancel_dict[k] for k in send_list}
        send_dict["event_type"] = "QUOTE_CANCEL"
        send_dict["event_time"] = datetime.now()
        rds.insert(**cfg.hibiscus_db, table="transactions", data=send_dict)
        logger.info(f"Cancelled standing quote for counterparty {cpty_id}.")
        return 'I\'ve cancelled your outstanding quote.'
    else:
        logger.info(f"Attempted a cancel on {cpty_id}, no such quote found.")
        return 'You have no active quotes, no action taken.'


async def modify_ticket(config: Dict):
    """
    Updates the state of the quote with any of the original ticket parameters. 
    If the counterparty does not have an existing ticket, we ask them to ask
    for a new ticket with the full JSON.

    Accepts a JSON of the following format in the "response" field:
    {
        "identifier": "..",
        "side": "..",
        "size": "..",
        "quote_method": ".."
    }
    All fields are optional, but at least one must be provided.
    "identifier" must be a valid ISIN, CUSIP, SEDOL, or recognized ticker symbol.
    "side" must be "BUY", "SELL", or "2WAY" and must reflect the trade action being taken by the market maker, i.e. BUY means the market maker is buying.
    "size" must be either a quantity of the security or notional value to trade, expressed as an integer.
    "quote_method" must be the reference quote type the client would like to execute against. Accepted values are RISK, NAV, TWAP, VWAP, and GMOC.

    After updating state, the client will see a new quote refreshed within 5 seconds.
    """

    telegram_id = config["client_id"]
    cpty_id = rds.run(
        **cfg.hibiscus_db,
        query=f"SELECT counterparty_id "
              f"FROM hibiscus.telegram_users "
              f"WHERE telegram_id = '{telegram_id}'",
        return_dict=True
    )[0].get("counterparty_id")

    if not cpty_id:
        return 'User not registered with TAM. Please use /register to be onboarded.'
    elif cpty_id not in QUOTES:
        return 'You do not have any active quote requests. Please provide a new RFQ.'
    else:
        for k, v in config.items():
            if k in RESPONSE_FIELD_MAP:
                QUOTES[cpty_id][RESPONSE_FIELD_MAP[k]] = v

        get_quote(QUOTES[cpty_id])
        quote_dict = QUOTES[cpty_id]

        send_list = ["event_type", "event_time", "quote_side", "quote_type", "counterparty_id", "user_id",
                     "security", "figi", "quantity", "bid_offset", "bid_quote", "ask_offset", "ask_quote"]
        send_dict = {k: quote_dict[k] for k in send_list}
        send_dict["event_type"] = "QUOTE"
        send_dict["event_time"] = datetime.now()
        rds.insert(**cfg.hibiscus_db, table="transactions", data=send_dict)
        QUOTES[cpty_id] = quote_dict

        if quote_dict['quote_side'] == "2WAY":
            return 'We\'re at {} / {} for size {} of {}'.format(
                quote_dict['bid_offset'], quote_dict['ask_offset'], quote_dict['quantity'], quote_dict['identifier'])
        elif quote_dict['quote_side'] == "BID":
            return 'We\'re at {} to buy {} of {} from you'.format(
                quote_dict['bid_offset'], quote_dict['quantity'], quote_dict['identifier'])
        elif quote_dict['quote_side'] == "OFFER":
            return 'We\'re at {} to sell {} of {} to you'.format(
                quote_dict['ask_offset'], quote_dict['quantity'], quote_dict['identifier'])


async def lock_trade(config: Dict):
    """
    This method locks the counterparty's standing quote and shows them buttons to confirm a BUY or
    SELL. If the standing quote is 2-way, the client will see 2 buttons, one for BUY and one for
    SELL. If the quote is one-sided, the client only sees the button for that side. All trades will occur
    at the size being quoted; if the counterparty would like a new size they must request an update to the
    quote to reflect the new size first and wait to receive a new quote to trade against.
    """

    telegram_id = config["client_id"]
    cpty_id = rds.run(
        **cfg.hibiscus_db,
        query=f"SELECT counterparty_id "
              f"FROM hibiscus.telegram_users "
              f"WHERE telegram_id = '{telegram_id}'",
        return_dict=True
    )[0].get("counterparty_id")

    if cpty_id not in QUOTES:
        return "You have no standing quotes. No trade to confirm."
    else:
        quote_dict = QUOTES[cpty_id]
        logger.debug(f"Sending trade option buttons to client.")
        if quote_dict['quote_side'] == "2WAY":
            button_list = [
                InlineKeyboardButton("BUY", callback_data="BUY"),
                InlineKeyboardButton("SELL", callback_data="SELL"),
                InlineKeyboardButton("CANCEL", callback_data="CANCEL"),
            ]
            reply_markup = InlineKeyboardMarkup(
                build_menu(button_list, n_cols=3))
            text = 'Please confirm which side you would like to trade: current quote stands at {} / {} for size {} of {}'.format(
                quote_dict['bid_offset'], quote_dict['ask_offset'], quote_dict['quantity'], quote_dict['identifier'])
            return text, reply_markup
        elif quote_dict['quote_side'] == "BID":
            button_list = [
                InlineKeyboardButton("SELL", callback_data="SELL"),
                InlineKeyboardButton("CANCEL", callback_data="CANCEL"),
            ]
            reply_markup = InlineKeyboardMarkup(
                build_menu(button_list, n_cols=2))
            text = 'Please confirm you would like to sell {} {} at {}'.format(
                quote_dict['quantity'], quote_dict['identifier'], quote_dict['bid_offset'])
            return text, reply_markup
        elif quote_dict['quote_side'] == "OFFER":
            button_list = [
                InlineKeyboardButton("BUY", callback_data="BUY"),
                InlineKeyboardButton("CANCEL", callback_data="CANCEL"),
            ]
            reply_markup = InlineKeyboardMarkup(
                build_menu(button_list, n_cols=2))
            text = 'Please confirm you would like to buy {} of {} at {}'.format(
                quote_dict['quantity'], quote_dict['identifier'], quote_dict['ask_offset'])
            return text, reply_markup


async def confirm_trade(update, context):

    logging.debug(update)

    telegram_id = update.callback_query.from_user.id
    cpty_id = rds.run(
        **cfg.hibiscus_db,
        query=f"SELECT counterparty_id "
              f"FROM hibiscus.telegram_users "
              f"WHERE telegram_id = '{telegram_id}'",
        return_dict=True
    )[0].get("counterparty_id")

    query = update.callback_query
    query.answer()

    if query.data == "CANCEL":
        await context.bot.send_message(text="Cancelled trade, but quote is still active.", chat_id=telegram_id)
        return 0

    trade_dict = QUOTES.pop(cpty_id)
    if trade_dict:
        send_list = ["event_type", "event_time", "quote_type", "counterparty_id", "user_id",
                     "security", "figi", "quantity", "bid_offset", "bid_quote", "ask_offset", "ask_quote", "quote_side"]
        send_dict = {k: trade_dict[k] for k in send_list}
        send_dict["event_type"] = "TRADE"
        send_dict["event_time"] = datetime.now()
        if query.data == "SELL":
            send_dict["side"] = "BUY"
            price = round(send_dict['bid_quote'] + send_dict['bid_offset'], 2)
        else:
            send_dict["side"] = "SELL"
            price = round(send_dict['ask_quote'] + send_dict['ask_offset'], 2)
        send_dict['px'] = price
        rds.insert(**cfg.hibiscus_db, table="transactions", data=send_dict)
        logger.info(f"Executed trade and logged transaction.")
        await context.bot.send_message(text=f"You've executed {query.data} {send_dict['quantity']} at {price}. You can see a trade confirmation in your trade history report.", chat_id=telegram_id)
        return 0
    else:
        logger.info(f"Tried to execute a trade when a quote did not exist.")
        await context.bot.send_message(text=f"No trade possible, there is no outstanding quote.", chat_id=telegram_id)
        return 0


async def get_trade_history(config):
    """ Gives the client a PDF that shows all trades the client has executed with TAM.
    """

    telegram_id = config["client_id"]

    q1 = f"""
    select tu.counterparty_id, 
      tu.first_name, 
      tu.last_name, 
      cp.bd, 
      address1, 
      address2 
    from telegram_users tu join counterparty cp
    on tu.counterparty_id=cp.id
    where tu.telegram_id={telegram_id}
    """
    data = rds.run(**cfg.hibiscus_db, query=q1, return_dict=True)

    q2 = f"""
    select 
    date_format(event_time, "%Y-%m-%d %H:%i:%S"),
    `security`,
    case when side="BUY" then "SELL" when side="SELL" then "BUY" else NULL end as side,
    quantity,
    px
    from telegram_users tu join transactions t
    on tu.telegram_id = t.user_id and tu.counterparty_id=t.counterparty_id
    where tu.telegram_id={telegram_id}
    AND event_type='TRADE'
    AND t.event_time>=curdate()
    order by event_time desc;
    """

    t = rds.run(**cfg.hibiscus_db, query=q2, return_dict=True)
    trades = [_justify_row(trade.values()) for trade in t]

    filename = f"{uuid4().hex}.pdf"
    if os.name == 'nt':
        path_to_use = cfg.pdf_filepath_windows + filename
    else:
        path_to_use = cfg.pdf_filepath_aws + filename
    logger.info(f"Generating pdf here: {path_to_use}")

    _generate_pdf(f'{path_to_use}', data, trades)
    _generate_log('HISTORY', config["client_id"], config["msg"])

    txt = """Here's your transactions executed today!"""
    return path_to_use + "|" + txt


async def cancel(update, context):
    """Registers a real broker dealer to the user for demo purposes."""

    logger.info("User %s canceled the conversation",
                update.message.from_user.id)
    _generate_log('GOODBYE', update.message.from_user.id)

    txt = f"""Goodbye, {update.message.from_user.first_name}! Talk soon."""
    await context.bot.send_message(text=txt, chat_id=update.message.from_user.id)

    # write a quote cancel and then wipe quote if they have an outstanding quote
    cpty_id = rds.run(
        **cfg.hibiscus_db,
        query=f"SELECT counterparty_id "
              f"FROM hibiscus.telegram_users "
              f"WHERE telegram_id = '{update.message.from_user.id}'",
        return_dict=True
    )[0].get("counterparty_id")

    if QUOTES.pop(cpty_id):
        cancel_dict = QUOTES.get(cpty_id)
        cancel_dict["event_type"] = "QUOTE_CANCEL"
        cancel_dict["event_time"] = datetime.now()
        rds.insert(**cfg.hibiscus_db, table="transactions", data=cancel_dict)

    return ConversationHandler.END


async def get_last_trade(config):
    """
    Shows the client the details of their last trade with TAM. 
    """

    query = f"""
    select 
    concat(event_type, "\n",
    date_format(event_time, "%Y-%m-%d %H:%i:%S"),
    "\n",
    `security`,
    "\n",
    side,
    " ",
    quantity,
    "@",
    px) as val
    from telegram_users tu join (select event_time, event_type, user_id, counterparty_id, `security`, case when side="BUY" then "SELL" when side="SELL" then "BUY" else NULL end as side, quantity, px from transactions) t
    on tu.telegram_id = t.user_id and tu.counterparty_id=t.counterparty_id
    WHERE tu.telegram_id= {config["client_id"]}
    order by event_time desc
    limit 1;
    """

    logger.debug(f"get last trade query {query}")

    # log this
    _generate_log("LAST_TRADE", config["client_id"], config["msg"])

    result = rds.run(**cfg.hibiscus_db, query=query, return_dict=True)
    if len(result) > 0:
        txt = '\n'.join([str(i) for i in result[0].values()])
    else:
        txt = """Hm, now where did I put your last trade..."""
    return txt


def _generate_pdf(path, data, trades):
    my_canvas = canvas.Canvas(path, pagesize=letter)
    my_canvas.setLineWidth(.3)
    my_canvas.setFont('Courier', 11)
    my_canvas.drawString(
        30, 750, f"""{data[0]['first_name']} {data[0]['last_name']}""")
    my_canvas.drawString(30, 735, data[0]['bd'])
    if data[0]['address2'] is not None:
        my_canvas.drawString(30, 720, data[0]['address2'])
    else:
        my_canvas.drawString(30, 720, data[0]['address1'])
    my_canvas.drawString(
        500, 750, datetime.now().strftime("%Y-%m-%d"))
    my_canvas.drawString(30, 690, "TRADES")
    my_canvas.line(30, 685, 580, 685)

    row = 670

    for trade in trades:
        my_canvas.drawString(30, row, trade)
        row -= 15

    my_canvas.save()


def _justify_row(row):
    row = list(row)
    output = [str(row[0]).ljust(25), str(row[1]).ljust(30)]
    for col in row[2:]:
        output.append(str(col).ljust(10))
    return ''.join(output)


def _generate_log(event_type: str, id, msg=None):
    tlog = {'event_type': event_type,
            'event_time': datetime.now(),
            'user_id': id,
            'user_msg': msg}

    try:
        rds.upsert(**cfg.hibiscus_db, table='transactions', data=tlog)
    except:
        # sometimes databases just fail could be queued to redis but whatevs
        pass

    return tlog


async def help(context):

    user_message = context['user_message']

    conversation_messages = [
        {'role': 'system', 'content': CONVERSATION_PROMPT},
        {'role': 'user', 'content': user_message}
    ]

    response = get_completion_from_messages(
        conversation_messages, temperature=0)
    return response


def build_menu(
    buttons: List[InlineKeyboardButton],
    n_cols: int,
    header_buttons: Union[InlineKeyboardButton,
                          List[InlineKeyboardButton]] = None,
    footer_buttons: Union[InlineKeyboardButton,
                          List[InlineKeyboardButton]] = None
) -> List[List[InlineKeyboardButton]]:
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, header_buttons if isinstance(
            header_buttons, list) else [header_buttons])
    if footer_buttons:
        menu.append(footer_buttons if isinstance(
            footer_buttons, list) else [footer_buttons])
    return menu
