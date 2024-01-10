from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter
import numpy as np
from numpy.typing import NDArray, ArrayLike
from openai import OpenAI
from os import environ as env
from py_dotenv import read_dotenv
import json
from hashlib import sha256
from datetime import datetime as dt


try:
    read_dotenv(".env")
except:
    print("Couldn't find .env file, must be running in docker so ima use env vars")

client = OpenAI(api_key=env["OPENAIKEY"], organization=env["OPENAIORG"])

UINT64_MAX = 2**64 - 1
TIMEOUT = 6

try:
    with open("data/text.json", "r") as f:
        text_db: dict[str, str] = json.load(f)
except:
    text_db = {}

try:
    with open("data/texthases.npy", "rb") as f:
        text_hashes: NDArray[np.uint64] = np.load(f)
except:
    text_hashes = np.array([], dtype=np.uint64)

try:
    with open("data/embeds.npy", "rb") as f:
        embeds: NDArray[np.float64] = np.load(f)
except:
    embeds = np.zeros((1, 1536), dtype=np.float64)


class Handler(BaseHTTPRequestHandler):
    text_db:dict[str,str] = text_db
    text_hashes: NDArray[np.uint64]
    page_ids: NDArray[np.uint32]
    embeds: NDArray

    # def log_message(self, *_, **__) -> None:
    #    return

    @staticmethod
    def unit_l2_normalization(vector: NDArray[np.float64]):
        """Normalize a vector using L2 normalization."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @staticmethod
    def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return 1 - np.dot(a, b)

    @staticmethod
    def hashedEmbed(embed: NDArray[np.float64]) -> str:
        return sha256(embed.__str__().encode("utf-8")).hexdigest()

    @staticmethod
    def textHash(text: str) -> int:
        return min(int(sha256(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16], 16), UINT64_MAX)

    def lookupEmbed(self, hashedEmbed: str) -> str | None:
        try:
            return Handler.text_db[hashedEmbed]
        except KeyError:
            return ""

    def createEmbedding(self, text: str) -> NDArray[np.float64]:
        return self.unit_l2_normalization(
            np.array(
                client.embeddings.create(input=text, model=env["EMBEDMODEL"])
                .data[0]
                .embedding,
                dtype=np.float64,
            )
        )

    def query(self, text: str):
        queryEmbed = self.createEmbedding(text)
        temp: dict[float, str] = {}
        for embed in Handler.embeds:
            embed: NDArray[np.float64]
            distance = self.distance(queryEmbed, embed)
            lookedup = self.lookupEmbed(self.hashedEmbed(embed))
            if lookedup:
                temp[distance] = lookedup
        return [
            i
            for _, i in sorted(temp.items(), key=lambda x: x[0])
            if not i.startswith("== ")
        ][:3]

    def send(self, code: int, data: dict):
        to_send = json.dumps(data)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(to_send)))
        self.end_headers()
        self.wfile.write(to_send.encode("utf-8"))

    def json(self) -> dict:
        # content_len = self.headers.get("Content-Length", 0)
        length = int(self.headers.get("Content-Length", 0))
        out = self.rfile.read(length)
        if length == 0:
            raise ValueError("No data")
        return json.loads(out.decode("utf-8"))

    def do_POST(self):
        try:
            if self.path == "/add":
                items = 0
                json = self.json()
                texts: set[str] = set(sorted(json["texts"], key=lambda x: len(x)))

                st = perf_counter()
                for text in texts:
                    # print("")
                    if perf_counter() - st > TIMEOUT:
                        break
                    # print("finished perf time check")

                    if text.startswith("== See also ==\n") or text.endswith("=="):
                        continue
                    # print("finished see also check")

                    texthash = self.textHash(text)
                    # print("finished hash gen")
                    if texthash in Handler.text_hashes:
                        continue
                    # print("finished hash check")

                    textEmbed = self.createEmbedding(text)
                    # print("finished embed gen")
                    hashed = self.hashedEmbed(textEmbed)
                    # print("finished embed hash gen")
                    if not self.lookupEmbed(hashedEmbed=hashed):
                        Handler.text_db[hashed] = text
                        Handler.embeds = np.append(Handler.embeds, [textEmbed], axis=0)
                        Handler.text_hashes: NDArray[np.uint64] = np.append(
                            Handler.text_hashes, [texthash]
                        )

                    items += 1
                print(f"Added {items} items")
                self.send(200, {"success": True, "items": items})

            elif self.path == "/query":
                json = self.json()
                # print(json)
                items = self.query(json["text"])
                self.send(200, {"success": True, "items": items})

            else:
                self.send(404, {"success": False, "error": "not found"})
                return
        except Exception as e:
            self.send(500, {"success": False, "error": str(e)})
            raise e


def main():
    # if you run this locally, you might want to change it to a
    # specific interface. im using * because its in a docker container
    Handler.text_db = text_db
    Handler.embeds = embeds
    Handler.text_hashes = text_hashes
    Handler.protocol_version = "HTTP/1.1"
    Handler.timeout = 50

    def do_save():
        # save everything
        with open("data/text.json", "w") as f:
            json.dump(Handler.text_db, f)
        with open("data/embeds.npy", "wb") as f:
            np.save(f, Handler.embeds)
        with open("data/texthashes.npy", "wb") as f:
            np.save(f, Handler.text_hashes)

    class NewServer(ThreadingHTTPServer):
        def service_actions(self):
            # weekly delete--nested for performance
            if dt.today().isoweekday() == 0:
                if dt.today().hour == 0:
                    if dt.today().minute == 0:
                        if dt.today().second == 0 and dt.today().microsecond in range(
                            100000
                        ):
                            Handler.text_db = {}
                            Handler.embeds = np.zeros((1, 1536), dtype=np.float64)
                            Handler.text_hashes = np.array([], dtype=np.uint64)
            do_save()
            return super().service_actions()

    server = NewServer(("0.0.0.0", 4211), Handler)

    try:
        print("Hosted on http://localhost:4211")
        server.serve_forever()
    except:
        try:
            server.server_close()
        except:
            pass
        do_save()


if __name__ == "__main__":
    main()
