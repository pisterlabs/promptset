# %%
import threading
import multiprocessing
import time
import random
import math
import abc
from collections import namedtuple
import os
import itertools
import dotenv
from operator import itemgetter
from datetime import datetime
import csv
import re
import requests
import json

import chess
import openai
from stockfish import Stockfish as _Stockfish
from constants import STOCKFISH_PATH, WIN_CUTOFF

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Stockfish implements this class
class APIEngine(abc.ABC):
    @abc.abstractmethod
    def get_best_move(self):
        "returns UCI string"
        pass

    @abc.abstractmethod
    def set_position(self, moves):
        pass

    @abc.abstractmethod
    def get_elo(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__} named: {self.name}"


class Stockfish(_Stockfish, APIEngine):
    name = None

    def __init__(self, name=None, *vargs, **kwargs):
        _Stockfish.__init__(self, STOCKFISH_PATH, *vargs, **kwargs)
        self.name = name

    def __str__(self):
        elo = self.get_parameters()["UCI_Elo"]
        return super().__str__() + f" elo: {elo}"

    def get_elo(self):
        return self.get_parameters()["UCI_Elo"]

    # It can be faster to not use this because the games are shorter
    # was faster with 100ms than not
    # def get_best_move(self):
    #    """Limits time per move, may change elo.
    #    delete method if want raw elo
    #    """
    #    return _Stockfish.get_best_move_time(self, 250)


class OpenAI(APIEngine):
    name = None

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.san_moves = []
        self.uci_moves = []
        self.board = chess.Board()
        self.elo_est = 0  # metadata only

    def get_elo(self):
        return self.elo_est

    def __str__(self):
        return super().__str__() + f" model: {self.model}"

    def set_position(self, uci_moves):
        # Sticking with UCI moves for now, but prompt uses SAN
        self.uci_moves = [*uci_moves]
        san_moves = []
        self.board.reset()
        for m in uci_moves:
            san_moves += [self.board.san(chess.Move.from_uci(m))]
            self.board.push_uci(m)
        self.san_moves = san_moves

    def _make_prompt(self):
        prompt = (
            #    """Complete the remainder of this game: `\n"""
            """[Event "FIDE World Championship match 2024"]\n"""
            """[Site "Los Angeles, USA"]\n"""
            """[Date "2024.11.11"]\n"""
            """[Round "13"]\n"""
            """[White "Carlsen, Magnus (NOR)"]\n"""
            """[Black "Nepomniachtchi, Ian (RUS)"]\n"""
            """[Result "0-1"]\n"""
            """[WhiteElo "2882"]\n"""
            """[White Title "GM"]\n"""
            """[BlackElo "2812"]\n"""
            """[BlackTitle "GM"]\n"""
            """[TimeControl "40/7200:20/3600:900+30"]\n"""
            """[UTCDate "2024.11.11"]\n"""
            """[UTCTime "12:00:00"]\n"""
            """[Variant "Standard"]\n"""
        )
        prompt = (
            '[Event "FIDE World Cup 2023"]\n[Site "Baku AZE"]\n[Date "2023.08.23"]\n[EventDate'
            ' "2021.07.30"]\n[Round "8.2"]\n[Result "1/2-1/2"]\n[White "Rameshbabu'
            ' Praggnanandhaa"]\n[Black "Magnus Carlsen"]\n[ECO "C48"]\n[WhiteElo "2690"]\n[BlackElo'
            ' "2835"]\n[PlyCount "60"]\n\n'
        )
        prompt = """[Event ""FIDE World Championship Match 2024""]
[Site ""Los Angeles, USA""]
[Date ""2024.12.01""]
[Round ""5""]
[White ""Carlsen, Magnus""]
[Black ""Nepomniachtchi, Ian""]
[Result ""1-0""]
[WhiteElo ""2885""]
[White Title ""GM""]
[WhiteFideId ""1503014""]
[BlackElo ""2812""]
[BlackTitle ""GM""]
[BlackFideId ""4168119""]
[TimeControl ""40/7200:20/3600:900+30""]
[UTCDate ""2024.11.27""]
[UTCTime ""09:01:25""]
[Variant ""Standard""]\n"""

        prompt += "\n".join(
            [
                f"{i+1}. {wm} {bm}"
                for i, (wm, bm) in enumerate(
                    itertools.zip_longest(self.san_moves[0::2], self.san_moves[1::2], fillvalue="")
                )
            ]
        )
        if len(self.san_moves) % 2 == 0:
            prompt += f"\n{len(self.san_moves)//2+1}. " if len(self.san_moves) > 0 else "\n1. "
        # Parrot chess
        pgn = """[Event "FIDE World Championship Match 2024"]
[Site "Los Angeles, USA"]
[Date "2024.12.01"]
[Round "5"]
[White "Carlsen, Magnus"]
[Black "Nepomniachtchi, Ian"]
[Result "1-0"]
[WhiteElo "2885"]
[WhiteTitle "GM"]
[WhiteFideId "1503014"]
[BlackElo "2812"]
[BlackTitle "GM"]
[BlackFideId "4168119"]
[TimeControl "40/7200:20/3600:900+30"]
[UTCDate "2024.11.27"]
[UTCTime "09:01:25"]
[Variant "Standard"]

1."""

        for n, m in enumerate(self.san_moves):
            if n % 2 == 0 and n > 0:
                pgn += f" {n//2+1}."
            pgn += f" {m}"
        return pgn

    def get_best_move(self):
        prompt = self._make_prompt()
        # print(f"prompt: `{prompt}`")
        suggestions = []
        i = 0
        errors = 0
        while i < 2 and errors < 3:
            try:
                response = openai.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    temperature = min(0.1 + (i / 8), 2) #, min(0.8 + (i / 8), 2),
                    max_tokens=6,  # longest san moves are 6 tokens: dxe8=R#
                    stop=[
                        ".",
                        "1-0",
                        "0-1",
                        "1/2",
                    ],  # sometimes space is the first character generated
                    n=5,  # prompt is 175 tokens, cheaper to get all suggestions at once
                )
            except Exception as e:
                print("API request error", e)
                time.sleep(3 + 2**errors)
                errors += 1
                continue
            # print(response)
            texts = list(
                set(dict.fromkeys(map(itemgetter("text"), response.choices)))
            )  # Dict's perserve order in py>=3.7
            for text in texts:
                san_move = text.strip().split(" ")[0].split("\n")[0].strip()
                try:
                    try:
                        uci_move = self.board.parse_san(san_move).uci()
                    except chess.AmbiguousMoveError as e:
                        print(f"WARN Ambigious '{san_move}'. Null contest? {e}")
                        color = chess.WHITE if len(self.uci_moves) % 2 == 0 else chess.BLACK
                        if len(san_move) == 2:
                            piece = chess.PAWN
                            end = san_move[:2]
                            if "x" in san_move:
                                # "exd6 e.p." for en passant capture
                                end = san_move.replace("x", "")[1:3]
                        else:
                            piece = next(
                                p
                                for p in [
                                    chess.PAWN,
                                    chess.KNIGHT,
                                    chess.BISHOP,
                                    chess.ROOK,
                                    chess.QUEEN,
                                    chess.KING,
                                ]
                                if chess.piece_symbol(p) == san_move[0].lower()
                            )
                            end = san_move.replace("x", "")[1:3]
                        start_squares = [
                            chess.square_name(p) for p in self.board.pieces(piece, color)
                        ]
                        uci_move = next(
                            (
                                f"{start}{end}"
                                for start in start_squares
                                if chess.Move.from_uci(f"{start}{end}") in self.board.legal_moves
                            ),
                            None,
                        )
                        if uci_move is None:
                            uci_move = next(
                                m.uci()
                                for m in self.board.legal_moves
                                if m.uci()[:2] in start_squares
                            )
                    assert self.board.is_legal(self.board.parse_uci(uci_move))
                    return uci_move
                except Exception as e:
                    print(f"{e}, '{text}', '{san_move}'")
                    suggestions += [text]
            i += 1
        return f"No valid suggestions: {'|'.join(suggestions)}"


class Manual(OpenAI):
    def __init__(self, name, model):
        super().__init__(name, model)

    def __str__(self):
        return super().__str__() + f" Human Playing named {self.name}"

    def get_best_move(self):
        print(self.board)
        print("Enter move: ")
        while True:
            try:
                m = input()
                if m == "q":
                    return None
                self.board.push_san(m)
                return self.board.peek().uci()
            except Exception as e:
                print(e)
                print("Enter move: ")


class ParrotChess(OpenAI):
    """Playing against model https://parrotchess.com/
    Should be black
    """

    url = "https://parrotchess.com/move"

    headers = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": "https://parrotchess.com",
        "Referer": "https://parrotchess.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.47"
        ),
        "sec-ch-ua": '"Microsoft Edge";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }

    def __init__(self):
        super().__init__("ParrotChess", "webmodel-gpt-3.5-turbo-instruct")

    def get_best_move(self):
        payload = json.dumps({"moves": self.uci_moves})

        # Make the request
        for i in range(3):
            response = requests.post(ParrotChess.url, headers=ParrotChess.headers, data=payload)
            try:
                return response.json()["gptMove"]
            except Exception as e:
                print(e)
                print(response)
                time.sleep(2 + 2**i)
        return None


def new_elos(elo1, elo2, result, k=24):
    """result in terms of player 1
    k: a factor to determine how much elo changes, higher changes more quickly
    """
    try:
        num_matches = len(result)
        result = sum(result)
    except:
        num_matches = 1
    expected = num_matches / (1 + math.pow(10, (elo1 - elo2) / 400))
    elo1 += k * (result - expected)
    elo2 += k * (expected - result)
    return elo1, elo2


GameResults = namedtuple(
    "game_results",
    [
        "white",
        "black",
        "result",
        "time",
        "illegal_move",
        "moves",
        "fen",
    ],
)

StoreResults = namedtuple(
    "store_results",
    [
        *GameResults._fields,
        "white_elo",
        "black_elo",
        "eval",  # {'type': 'mate', 'value': nmoves_till_mate} or {'type': 'cp', 'value': centipawns_in_whites_favor}
    ],
)


def engines_play(white, black, uci_moves=None):
    """2 engines play against each other, e1 is white, e2 black
    Args:
        white (APIEngine):
        black (APIEngine):
        uci_moves (list): optional list of starting uci strings, ["e2e4"]
    Returns results namedtuple where result is the value for white
    """
    board = chess.Board()

    if uci_moves is None:
        uci_moves = []
    else:
        uci_moves = [*uci_moves]
    for m in uci_moves:
        board.push_uci(m)
    white_first = len(uci_moves) % 2
    white.set_position(uci_moves)
    black.set_position(uci_moves)
    t = time.perf_counter()
    illegal_move = None
    for i in range(6000):  # max possible moves is 5.9k
        # print(board, "\n")
        turn = white if (i % 2) == white_first else black
        turn.set_position(uci_moves)
        m = turn.get_best_move()
        # print(m, i)
        try:
            board.push_uci(m)
        except Exception as e:
            print(e)
            illegal_move = m
        if m is None or illegal_move is not None:
            result = 0 if turn == white else 1
            illegal_move = m
            break
        uci_moves += [m]

        # result depends whose turn it is
        if board.outcome() is not None:
            result = board.outcome().result().split("-")[0]
            if result == "1/2":
                result = 0.5
            else:
                result = float(result)
            break
        elif board.can_claim_draw():  # outcome() uses 5 rep and 75 moves
            result = 0.5
            break

    t = time.perf_counter() - t
    return GameResults(
        white.name,
        black.name,
        result,
        t,
        illegal_move,
        uci_moves,
        board.fen(),
    )


def make_engines(sf_elo=1200, model="gpt-3.5-turbo-instruct"):
    sf = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 512, "UCI_Elo": sf_elo})
    oa = OpenAI(model, model)
    oa.elo_est = sf_elo
    return (sf, oa)


def play_sf_oa(sf, oa):
    sf_white = bool(random.randint(0, 1))
    white, black = (sf, oa) if sf_white else (oa, sf)
    try:
        gr = engines_play(white, black)
    except Exception as e:
        print(e)
        return None

    print(chess.Board(gr.fen))
    eval = None
    if gr.illegal_move is not None:
        sf_elo = sf.get_parameters()["UCI_Elo"]
        sf.set_skill_level(20)
        sf.set_fen_position(gr.fen)
        eval = sf.get_evaluation()
        sf.set_elo_rating(sf_elo)
    r = StoreResults(*gr, white.get_elo(), black.get_elo(), eval)
    return tuple(r)


def play_sf_oa_ts(elo, model):
    # sf takes <1sec to init, but has locks to executable
    sf, oa = make_engines(elo, model)
    return play_sf_oa(sf, oa)


def play_sf_pc_ts(sf_elo, *vargs, **kwargs):
    sf = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 512, "UCI_Elo": sf_elo})
    pc = ParrotChess()
    pc.elo_est = sf_elo
    return play_sf_oa(sf, pc)


# sf_elo = 1200
# oa_elo = 1200
# sf = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 512})
# sf2 = Stockfish("weak_stockfish", parameters={"Threads": 6, "Hash": 128})
# referee = Stockfish("stockfish", parameters={"Threads": 6, "Hash": 1024, "Skill Level": 20})
# sf.set_elo_rating(sf_elo)
# sf2.set_elo_rating(sf_elo // 2)
# model = "gpt-3.5-turbo-instruct"
# oa = OpenAI(model, model)
# print(sf, sf2)

# moves = []
# for i in range(9):
#    sf.set_position(moves)
#    moves += [sf.get_best_move()]
# oa.set_position(moves)
# print(oa.get_best_move())


sf, oa = make_engines()
# m = Manual("human", "human")
# pc = ParrotChess()
# play(sf, m)
# play(sf, pc)
# print(play(sf, oa))
# print(oa._make_prompt())

oa.set_position(
    [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1c4",
        "g8f6",
        "d2d3",
        "f8c5",
    ]
)
print(oa._make_prompt())
# %%

if __name__ == "__main__":
    sf_elo = 1400
    oa_elo = 1400
    model = "gpt-3.5-turbo-instruct"
    all_results = []
    now = re.sub("\D+", "_", str(datetime.now()))
    NUM_CPU = 3
    NUM_GAMES = NUM_CPU * 3

    if False:  # parrot chess
        play = play_sf_pc_ts
        result_csv_path = f"results/results_pc_{now}.csv"
    else:  # local gpt
        play = play_sf_oa_ts
        result_csv_path = f"results/results_{now}.csv"

    print(result_csv_path)
    with open(result_csv_path, "a+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(StoreResults._fields)
        pool = multiprocessing.Pool(NUM_CPU)
        for _ in range(NUM_GAMES // NUM_CPU):
            results = [
                StoreResults(*r)
                for r in pool.starmap(play, [(sf_elo, model) for _ in range(NUM_CPU)])
            ]
            rs = []
            for r in results:
                if r is None:
                    print("game failed due to exception")
                    continue
                else:
                    all_results += [r]
                writer.writerow(r)
                if r.eval is None:
                    white_win = r.result
                elif r.eval["type"] == "mate":
                    white_win = int(
                        r.eval["value"] > 0 or (r.eval["value"] == 0 and len(r.moves) % 2 == 1)
                    )
                else:
                    if r.eval["value"] >= WIN_CUTOFF:
                        white_win = 1
                    elif r.eval["value"] <= -WIN_CUTOFF:
                        white_win = 0
                    else:
                        white_win = 0.5
                is_gpt = r.white != "stockfish"
                rs += [white_win if is_gpt else 1 - white_win]
            print([(i.white, i.result, i.eval) for i in results], rs)
            # oa_elo, sf_elo = new_elos(oa_elo, sf_elo, rs, k=50)
            print(results, "\n", rs, oa_elo, sf_elo)
            oa_elo += 200
            sf_elo = oa_elo
    print(all_results)
    print([(i.result, i.black_elo, i.eval, i.white) for i in all_results])
    print("\n\nwrote: ", result_csv_path)
# %%
