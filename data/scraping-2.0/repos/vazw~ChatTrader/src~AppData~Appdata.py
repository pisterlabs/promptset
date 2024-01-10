# The BEERWARE License (BEERWARE)
#
# Copyright (c) 2022 Author. All rights reserved.
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# vazw wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer or coffee in return

from datetime import datetime
import os
import openai
import sqlite3
import bcrypt
from dataclasses import dataclass

import mplfinance as mplf
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv

load_dotenv()

barsC = 1502


rcs = {
    "axes.labelcolor": "white",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.axisbelow": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "axes.titlesize": "xx-large",
    "axes.titleweight": "bold",
}


colors_candle = mplf.make_marketcolors(
    up="black",
    down="white",
    wick="white",
    edge="white",
    volume={"up": "green", "down": "red"},
)
style_candle = mplf.make_mpf_style(
    base_mpf_style="nightclouds",
    rc=rcs,
    y_on_right=False,
    marketcolors=colors_candle,
    figcolor="black",
    gridaxis="horizontal",
    facecolor="black",
)

BOTCOL = [
    "id",
    "symbol",
    "timeframe",
    "ATR",
    "ATR_m",
    "EMA",
    "subhag",
    "smooth",
    "RSI",
    "Andean",
    "Uselong",
    "Useshort",
    "UseTP",
    "UseTP2",
    "UseSL",
    "Tail_SL",
    "leverage",
    "Pivot",
    "RR1",
    "RR2",
    "TP1",
    "TP2",
    "Risk",
    "maxMargin",
    "hedge",
    "hedgeTF",
]

TA_TYPE = [
    "symbol",
    "timeframe",
    "ATR",
    "ATR_m",
    "EMA",
    "subhag",
    "smooth",
    "RSI",
    "Andean",
    "Pivot",
]

vxma_settings = {
    "id": "0",
    "symbol": "BTCUSDT",
    "timeframe": "6h",
    "ATR": 12,
    "ATR_m": 1.6,
    "EMA": 30,
    "subhag": 30,
    "smooth": 30,
    "RSI": 25,
    "Andean": 30,
    "Uselong": True,
    "Useshort": True,
    "UseTP": True,
    "UseTP2": False,
    "UseSL": True,
    "Tail_SL": True,
    "leverage": 50,
    "Pivot": 15,
    "RR1": 3,
    "RR2": 4.5,
    "TP1": 50,
    "TP2": 50,
    "Risk": "2$",
    "maxMargin": "10%",
    "hedge": True,
    "hedgeTF": "30m",
}

EGGS = [
    '"หุ้นดีขายออก หุ้นกระจอกเก็บไว้"',
    "หนีอินทนนท์ ปะเอเวอเรสต์",
    "หากเราเข้าออเดอร์ตามอารมณ์เราอาจจะเจอกับสถานการณ์อันอึดอัดนี้บ่อยๆนะ",
    "ซื้อที่ high ขายที่ low Show แต่ดอย ทะยอย cut loss",
    "มันเป็นฝันร้ายของเทรดเดอร์ ซึ่งมือใหม่ที่หวังรวยทางลัดโดยที่ไม่สนใจศึกษาเรียนรู้มักจะเจอปัญหานี้บ่อยๆ\
และมันก็จะนำไปสู่จุดจบแบบเดิมที่ไม่มีใครอยากให้เกิด",
    "คน คือทรัพย์สินที่มีค่ามากที่สุด…จงรักษาให้ดี",
    "ทุกการลงทุนมีความเสี่ยง แต่คุณควรมีเกณฑ์ ว่ารับได้แค่ไหน ถ้าพลาดก็ไม่เกิดปัญหาตามมา",
    "ปลาเห็นแต่เหยื่อล่อ ไม่เห็นขอเกี่ยว… นักลงทุนเห็นแต่กำไร ไม่เห็นความเสี่ยง..",
    "“ความล้มเหลว” ก็คือ “ความสำเร็จ” ถ้าหากเรา “เรียนรู้” จากมัน",
    "คุณสามารถทำมันให้ดีขึ้นได้อีกในทุกๆวัน",
    "หาเวลาพักบ้างนะคะ",
    "“ความล้มเหลว” ก็คือ “ความสำเร็จ” ถ้าหากเรา “เรียนรู้” จากมัน",
    "ปัญหาไม่ได้มีไว้เพื่อแก้ ปัญหามีไว้เพื่อเรียนรู้ ถ้าจะแก้ ให้แก้ที่เหตุของปัญหา",
    "รับกาแฟซักแก้วไหมคะนายท่าน ☕️",
]

vxma_settings_info = {
    "id": "0",
    "symbol": "คู่เทรด/เหรียญ",
    "timeframe": "Timeframe",
    "ATR": "ATR Length",
    "ATR_m": "ATR Mutiplier",
    "EMA": "EMA Length",
    "subhag": "Subhag Length",
    "smooth": "Subhag Smoother Length",
    "RSI": "RSI Length",
    "Andean": "Andean Osscrilator",
    "Uselong": "เปิด/ปิด Long",
    "Useshort": "เปิด/ปิด Short",
    "UseTP": "เปิด/ปิดใช้ TP",
    "UseTP2": "เปิด/ปิดใช้ TP2",
    "UseSL": "เปิด/ปิดใช้ SL",
    "Tail_SL": "เปิด/ปิดใช้ Tailing_SL",
    "leverage": "ตัวคูณ Leverage",
    "Pivot": "จำนวนแท่งย้อนหลัง",
    "RR1": "Risk:Reward สำหรับ TP",
    "RR2": "Risk:Reward สำหรับ TP2",
    "TP1": "จำนวน % การปิด Position ของ TP",
    "TP2": "จำนวน % การปิด Position ของ TP2",
    "Risk": 'ความเสี่ยงที่รับได้เป็นดอลลาร์ USD หากนำหน้าด้วย "%" จะเป็นการใช้เปอร์เซ็นเงินในพอร์ต\
เช่น %1 จะใช้เงินในพอร์ต 1%',
    "maxMargin": 'ขนาด Position สูงสุด ต่อ 1 Position เป็นดอลลาร์ USD หากนำหน้าด้วย "%"\
จะเป็นการใช้เปอร์เซ็นเงินในพอร์ตเช่น %1 จะใช้เงินในพอร์ต 1%',
    "hedge": "เปิด/ปิดใช้งาน Hedge Strategy",
    "hedgeTF": "Timeframe สำหรับกลยุทธ์ Hedging",
}


def bot_setting() -> pd.DataFrame:
    try:
        symbolist = pd.read_csv("bot_config.csv", index_col="id")
        return symbolist
    except Exception as e:
        print(e)
        return pd.DataFrame()


def config_setting():
    try:
        with sqlite3.connect("vxma.db", check_same_thread=False) as con:
            config = pd.read_sql("SELECT * FROM key", con=con)
        return config
    except Exception as e:
        print(e)
        return None


def cooking(id, pwd):
    try:
        pepper = f"{id}{pwd}!{barsC}vz{id}"
        bytePwd = pepper.encode("utf-8")
        Salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(bytePwd, Salt)
    except Exception as e:
        print(e)
        return None


def perf(id, pwd):
    hash1 = "X"
    try:
        with sqlite3.connect("vxma.db", check_same_thread=False) as con:
            bata = pd.read_sql("SELECT * FROM user", con=con)
        iid = bata["id"][0]
        if iid == id:
            hash1 = bata["pass"][0]
        egg = f"{id}{pwd}!{barsC}vz{id}"
        bytePwd = egg.encode("utf-8")
        return bcrypt.checkpw(bytePwd, hash1)
    except Exception as e:
        print(e)
        return None


def check_risk_str_float(risk: str, free_fiat: float) -> float:
    if "$" in risk:
        risk_ = float(risk.replace("$", ""))
    elif "%" in risk:
        percent = float(risk.replace("%", ""))
        risk_ = (percent / 100) * free_fiat
    else:
        risk_ = float(risk)
    return risk_


def caculate_margin(price, amt, lev) -> float:
    return float(price) * float(amt) / int(lev)


def max_margin_size(size: str, free_balance: float) -> float:
    if "$" in size:
        Max_Size = float(size.replace("$", ""))
    elif "%" in size:
        size = float(size.replace("%", ""))
        Max_Size = free_balance * (size / 100)
    else:
        Max_Size = float(size)
    return Max_Size


def remove_last_line_from_string(text: str) -> str:
    return text[: text.rfind("\n")]


class RiskManageTable:
    def __init__(self, symbolist, balance):
        self.symbol = symbolist["symbol"]
        self.quote = "BUSD" if self.symbol.endswith("BUSD") else "USDT"
        # if self.symbol[0:4] == "1000":
        #     self.symbol = self.symbol[4 : len(self.symbol)]
        self.timeframe = symbolist["timeframe"]
        self.use_long = self.check_bool(symbolist["Uselong"])
        self.use_short = self.check_bool(symbolist["Useshort"])
        self.use_tp_1 = self.check_bool(symbolist["UseTP"])
        self.use_tp_2 = self.check_bool(symbolist["UseTP2"])
        self.use_sl = self.check_bool(symbolist["UseSL"])
        self.use_tailing = self.check_bool(symbolist["Tail_SL"])
        self.free_balance = float(balance["free"][self.quote])
        self.max_size = max_margin_size(
            str(symbolist["maxMargin"]), self.free_balance
        )
        self.risk_size = str(symbolist["Risk"])
        self.tp_percent = symbolist["TP1"]
        self.tp_percent_2 = symbolist["TP2"]
        self.risk_reward_1 = symbolist["RR1"]
        self.risk_reward_2 = symbolist["RR2"]
        self.leverage = symbolist["leverage"]
        self.usehedge = self.check_bool(symbolist["hedge"])
        self.hedge_timeframe = symbolist["hedgeTF"]

    def check_bool(self, arg) -> bool:
        return True if str(arg).lower() == "true" else False


class DefaultRiskTable:
    def __init__(self, symbol: str, balance):
        self.symbol = symbol
        self.quote = "BUSD" if self.symbol.endswith("BUSD") else "USDT"
        # if self.symbol[0:4] == "1000":
        #     self.symbol = self.symbol[4 : len(self.symbol)]
        self.timeframe = "6h"
        self.use_long = True
        self.use_short = True
        self.use_tp_1 = True
        self.use_tp_2 = False
        self.use_sl = True
        self.use_tailing = True
        self.free_balance = float(balance["free"][self.quote])
        self.max_size = 20
        self.risk_size = "%5"
        self.tp_percent = 50
        self.tp_percent_2 = 50
        self.risk_reward_1 = 2
        self.risk_reward_2 = 3
        self.leverage = 10
        self.usehedge = False
        self.hedge_timeframe = "15m"


@dataclass
class Last_update:
    candle: str = "T -- ----------"
    balance: any = "--"
    status: str = "Starting"


@dataclass
class Timer:
    min_timewait: int = 1
    min_timeframe: str = "1m"
    last_closed: any = 0.0
    next_candle: any = 0.0
    get_time: bool = False


@dataclass
class PositionMode:
    dualSidePosition: bool = False
    Sside: str = "BOTH"
    Lside: str = "BOTH"


# ansi escape code
@dataclass
class ColorCS:
    CLS_SCREEN: str = "\033[2J\033[1;1H"  # cls + set top left
    CLS_LINE: str = "\033[0J"
    SHOW_CURSOR: str = "\033[?25h"
    HIDE_CURSOR: str = "\033[?25l"
    CRED: str = "\33[31m"
    CGREEN: str = "\33[32m"
    CYELLOW: str = "\33[33m"
    CEND: str = "\033[0m"
    CBOLD: str = "\33[1m"


@dataclass
class TATable:
    atr_p: int = 12
    atr_m: float = 1.6
    ema: int = 30
    linear: int = 30
    smooth: int = 30
    rsi: int = 25
    aol: int = 30
    pivot: int = 60


class AppConfig:
    """Get config for global App."""

    def __init__(self):
        config = config_setting()
        max_margin = "$10"
        MIN_BALANCE = "$50"
        if not config.empty:
            max_margin = str(config["freeB"][0])
            MIN_BALANCE = str(config["minB"][0])
            API_KEY = str(config["apikey"][0])
            API_SECRET = str(config["apisec"][0])
            LINE_TOKEN = str(config["notify"][0])
        else:
            API_KEY = ""
            API_SECRET = ""
            LINE_TOKEN = ""
        if "API_KEY" in os.environ:
            API_KEY = str(os.environ["API_KEY"])
            API_SECRET = str(os.environ["API_SECRET"])
            LINE_TOKEN = str(os.environ["Line_Notify_Token"])
        if MIN_BALANCE[0] == "$":
            self.min_balance = float(MIN_BALANCE[1 : len(MIN_BALANCE)])
        else:
            self.min_balance = float(MIN_BALANCE)
        if max_margin[0] == "$":
            self.max_margin = float(max_margin[1 : len(max_margin)])
        else:
            self.max_margin = float(max_margin)
        self.notify_token = LINE_TOKEN
        self.BNBCZ = {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
            "adjustForTimeDifference": True,
        }


# def notify_send(msg, sticker=None, package=None, image_path=None):
#     config = AppConfig()
#     notify = LineNotify(config.notify_token)
#     try:
#         if image_path is not None:
#             notify.send(msg, image_path=image_path)
#         elif sticker is not None:
#             notify.send(
#                 msg,
#                 sticker_id=sticker,
#                 package_id=package,
#             )
#         else:
#             notify.send(msg)
#     except Exception as e:
#         print(e)
#


def candle(df, symbol, tf):
    data = df.tail(100)
    titles = f"{symbol}_{tf}"
    try:
        vxma = mplf.make_addplot(
            data.vxma, secondary_y=False, color="yellow", linewidths=0.2
        )
        buy = mplf.make_addplot(
            data.buyPrice, secondary_y=False, color="green", scatter=True
        )
        sell = mplf.make_addplot(
            data.sellPrice, secondary_y=False, color="red", scatter=True
        )
        mplf.plot(
            data,
            type="candle",
            title=titles,
            addplot=[vxma, buy, sell],
            style=style_candle,
            volume=True,
            savefig="candle.png",
            tight_layout=True,
            figratio=(9, 9),
            datetime_format="%y/%b/%d %H:%M",
            xrotation=20,
        )
    except Exception as e:
        print(e)
        mplf.plot(
            data,
            type="candle",
            title=titles,
            style=style_candle,
            volume=True,
            savefig="candle.png",
            tight_layout=True,
            figratio=(9, 9),
            datetime_format="%y/%b/%d %H:%M",
            xrotation=20,
        )
    return "./candle.png"


def clearconsol():
    try:
        if os.name == "posix":
            os.system("clear")
        else:
            os.system("cls")
    except Exception as e:
        print(e)


def read_all_open_position_record():
    order_history = pd.read_csv("trades.csv")
    order_history = order_history[pd.isnull(order_history["ClosePrice"])]
    return order_history


def read_one_open_trade_record(
    symbol: str,
    timeframe: str,
    direction: str = "",
) -> pd.Series:
    order_history = pd.read_csv("trades.csv")
    position = None
    for id in order_history.index:
        if (
            order_history["Symbol"][id] == symbol
            and pd.isnull(order_history["ClosePrice"][id])
            and order_history["Position"][id] == direction
            and order_history["TF"][id] == timeframe
        ):
            position = order_history.loc[id,]
            break
    return position


def write_trade_record(
    timestamp: datetime,
    symbol: str,
    timeframe: str,
    amount: float,
    price: float,
    direction: str,
    tp: any = None,
    sl: any = None,
) -> None:
    # Create a dataframe from the input data
    df = pd.DataFrame(
        {
            "EntryTime": [timestamp],
            "ExitTime": [None],
            "Symbol": [symbol],
            "TF": [timeframe],
            "Position": [direction],
            "Amount": [amount],
            "EntryPrice": [price],
            "ClosePrice": [None],
            "TP": [tp],
            "SL": [sl],
            "PNL$": [None],
        }
    )

    # Append the dataframe to the CSV file
    # df.to_csv("trades.csv", index=False, header=True)
    df.to_csv("trades.csv", mode="a", index=False, header=False)


def write_tp_record(
    timestamp: datetime,
    symbol: str,
    timeframe: str,
    direction: str,
    price: float,
    amount: float,
    saved_position: pd.Series,
) -> None:
    # Create a dataframe from the input data
    order_history = pd.read_csv("trades.csv")
    for id in order_history.index:
        if (
            order_history["Symbol"][id] == symbol
            and pd.isnull(order_history["ClosePrice"][id])
            and order_history["Position"][id] == direction
            and order_history["TF"][id] == timeframe
        ):
            order_history["ExitTime"][id] = timestamp
            order_history["ClosePrice"][id] = price
            order_history["Amount"][id] = amount

            if order_history["Position"][id] == "Long":
                order_history["PNL$"][id] = round(
                    (
                        order_history["ClosePrice"][id]
                        - order_history["EntryPrice"][id]
                    )
                    * order_history["Amount"][id],
                    3,
                )
            else:
                order_history["PNL$"][id] = round(
                    (
                        order_history["EntryPrice"][id]
                        - order_history["ClosePrice"][id]
                    )
                    * order_history["Amount"][id],
                    3,
                )
    # rewrite the whole dataframe to the CSV file
    order_history.to_csv("trades.csv", index=False, header=True)

    df = pd.DataFrame(
        {
            "EntryTime": [saved_position["EntryTime"]],
            "ExitTime": [None],
            "Symbol": [symbol],
            "TF": [timeframe],
            "Position": [direction],
            "Amount": [float(saved_position["Amount"]) - amount],
            "EntryPrice": [price],
            "ClosePrice": [None],
            "TP": [saved_position["TP"]],
            "SL": [saved_position["SL"]],
            "PNL$": [None],
        }
    )

    # Append the dataframe to the CSV file
    # df.to_csv("trades.csv", index=False, header=True)
    df.to_csv("trades.csv", mode="a", index=False, header=False)


def edit_trade_record(
    timestamp: datetime,
    symbol: str,
    timeframe: str,
    direction: str,
    price: float,
    isSl: bool = False,
) -> None:
    # Create a dataframe from the input data
    order_history = pd.read_csv("trades.csv")
    for id in order_history.index:
        if (
            order_history["Symbol"][id] == symbol
            and pd.isnull(order_history["ClosePrice"][id])
            and order_history["Position"][id] == direction
            and order_history["TF"][id] == timeframe
        ):
            order_history["ExitTime"][id] = timestamp
            if isSl:
                order_history["ClosePrice"][id] = order_history["SL"][id]
            else:
                order_history["ClosePrice"][id] = price

            if order_history["Position"][id] == "Long":
                order_history["PNL$"][id] = round(
                    (
                        order_history["ClosePrice"][id]
                        - order_history["EntryPrice"][id]
                    )
                    * order_history["Amount"][id],
                    3,
                )
            else:
                order_history["PNL$"][id] = round(
                    (
                        order_history["EntryPrice"][id]
                        - order_history["ClosePrice"][id]
                    )
                    * order_history["Amount"][id],
                    3,
                )
    # rewrite the whole dataframe to the CSV file
    order_history.to_csv("trades.csv", index=False, header=True)


def edit_all_trade_record(
    timestamp: datetime,
    symbol: str,
    direction: str,
    price: float,
    isSl: bool = False,
) -> None:
    # Create a dataframe from the input data
    order_history = pd.read_csv("trades.csv")
    for id in order_history.index:
        if (
            order_history["Symbol"][id] == symbol
            and pd.isnull(order_history["ClosePrice"][id])
            and order_history["Position"][id] == direction
        ):
            order_history["ExitTime"][id] = timestamp
            if isSl:
                order_history["ClosePrice"][id] = order_history["SL"][id]
            else:
                order_history["ClosePrice"][id] = price

            if order_history["Position"][id] == "Long":
                order_history["PNL$"][id] = round(
                    (
                        order_history["ClosePrice"][id]
                        - order_history["EntryPrice"][id]
                    )
                    * order_history["Amount"][id],
                    3,
                )
            else:
                order_history["PNL$"][id] = round(
                    (
                        order_history["EntryPrice"][id]
                        - order_history["ClosePrice"][id]
                    )
                    * order_history["Amount"][id],
                    3,
                )
    # rewrite the whole dataframe to the CSV file
    order_history.to_csv("trades.csv", index=False, header=True)


REPLY_MARKUP = {
    "menu": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "💰เช็คกระเป๋าเงิน",
                    callback_data='{"M": "menu", "H": "CheckBalance"}',
                ),
                InlineKeyboardButton(
                    "💹เทรดมือ",
                    callback_data='{"M": "menuex", "H": "Trade"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "📈📉วิเคราะห์กราฟ",
                    callback_data='{"M": "menu", "H": "Analyser"}',
                ),
                InlineKeyboardButton(
                    "📊กำไร/ขาดทุน",
                    callback_data='{"M": "menu", "H": "PositionData"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "🤖บอท",
                    callback_data='{"M": "menu", "H": "BotSetting"}',
                ),
                # ย้ายไปยังเมนูบอท
                # InlineKeyboardButton(
                #     "⚙️ตั้งค่า",
                #     callback_data='{"M": "menu", "H": "apiSetting"}',
                # ),
                InlineKeyboardButton(
                    "❌ปิดเมนู",
                    callback_data='{"M": "menu", "H": "X"}',
                ),
            ],
        ]
    ),
    "fiat": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "USDT", callback_data='{"M": "fiat", "H": "USDT"}'
                ),
                InlineKeyboardButton(
                    "BUSD", callback_data='{"M": "fiat", "H": "BUSD"}'
                ),
                InlineKeyboardButton(
                    "ทั้งหมด", callback_data='{"M": "fiat", "H": "ALL"}'
                ),
            ],
            [
                InlineKeyboardButton(
                    "บังคับอัพเดตข้อมูล",
                    callback_data='{"M": "fiat", "H": "RE"}',
                ),
                InlineKeyboardButton(
                    "❌ กลับ", callback_data='{"M": "fiat", "H": "BACK"}'
                ),
            ],
        ]
    ),
    "secure": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "⚙️ตั้งค่า API",
                    callback_data='{"M": "secure", "H": "API"}',
                ),
            ],
            # [
            #     InlineKeyboardButton(
            #         "⚙️ตั้งค่ารหัสผ่าน",
            #         callback_data='{"M": "secure", "H": "PASS"}',
            #     ),
            # ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "secure", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "analyse": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "VXMA",
                    callback_data='{"M": "analyse", "H": "VXMA"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "analyse", "H": "BACK"}',
                )
            ],
        ]
    ),
    "order_type": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "MARKET",
                    callback_data='{"M": "order_type", "H": "MARKET"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "order_type", "H": "BACK"}',
                )
            ],
        ]
    ),
    "pnl": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "ℹ️ ดูรายละเอียด",
                    callback_data='{"M": "pnl", "H": "COINS"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ", callback_data='{"M": "pnl", "H": "BACK"}'
                ),
            ],
        ]
    ),
    "position_confirm_sl": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "position_confirm_sl", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "position_confirm_sl", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "position_confirm_tp": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "position_confirm_tp", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "position_confirm_tp", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "position_confirm_lev": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "position_confirm_lev", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "position_confirm_lev", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "vxma_settings_confirm": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "vxma_settings_confirm", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "vxma_settings_confirm", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "vxma_settings_confirm_del": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "vxma_settings_confirm_del", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "vxma_settings_confirm_del", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "vxma_settings_confirm_save": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "vxma_settings_confirm_save", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "vxma_settings_confirm_save", "H": "BACK"}',
                ),
            ],
        ]
    ),
    "vxma_settings_confirm_save_2": InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "✅ยืนยัน",
                    callback_data='{"M": "vxma_settings_confirm_save_2", "H": "OK"}',
                ),
            ],
            [
                InlineKeyboardButton(
                    "❌ กลับ",
                    callback_data='{"M": "vxma_settings_confirm_save_2", "H": "BACK"}',
                ),
            ],
        ]
    ),
}


openai.api_key = os.environ["OpenAiToken"]


def chat(messages):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
