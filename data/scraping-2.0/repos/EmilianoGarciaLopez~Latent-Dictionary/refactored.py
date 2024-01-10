import os
import pickle

import dash
import dash_bootstrap_components as dbc
import numpy as np
import openai
import plotly.graph_objects as go
import redis
import requests
from dash import html
from dash.dependencies import ALL, Input, Output, State
from dotenv import load_dotenv
from flask import Flask
from scipy.spatial import distance
from sklearn.decomposition import PCA

from layout import DEFAULT_WORD, app_layout, index_string

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")
openai.api_key = os.getenv("OPENAI_KEY")

OXFORD_3000_FILE = "oxford_3000_embeddings.pkl"


class EmbeddingsManager:
    def __init__(self):
        self.model = "text-embedding-ada-002"
        self.word_embeddings_map = self.get_or_create_embeddings()
        self.words = list(self.word_embeddings_map.keys())
        self.embeddings = np.array(list(self.word_embeddings_map.values()))
        self.pca = PCA(n_components=3)
        self.reduced_embeddings = self.pca.fit_transform(self.embeddings)

        self.redis_connection = redis.Redis(
            host="fly-embeddings-dictionary.upstash.io",
            port=6379,
            password="56c443d446f145f0a52ade710646a19d",
        )

    def save_embeddings_to_file(self, embeddings_map):
        with open(OXFORD_3000_FILE, "wb") as file:
            pickle.dump(embeddings_map, file)

    def load_embeddings_from_file(self):
        if os.path.exists(OXFORD_3000_FILE):
            with open(OXFORD_3000_FILE, "rb") as file:
                return pickle.load(file)
        return None

    def fetch_oxford_3000(self):
        url = "https://raw.githubusercontent.com/sapbmw/The-Oxford-3000/master/The_Oxford_3000.txt"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error {response.status_code}: Unable to fetch data")
            return []
        return [s.lower() for s in response.text.splitlines() if s.isalpha()]

    def get_or_create_embeddings(self):
        oxford_embeddings = self.load_embeddings_from_file()
        if oxford_embeddings:
            return oxford_embeddings
        word_embeddings_map = {}
        oxford_3000 = self.fetch_oxford_3000()
        batch_size = 1000
        n_batches = (len(oxford_3000) + batch_size - 1) // batch_size
        for i in range(n_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            words = oxford_3000[start:end]
            try:
                response = openai.Embedding.create(input=words, model=self.model)
                embeddings = [i["embedding"] for i in response["data"]]
                for word, embedding in zip(words, embeddings):
                    word_embeddings_map[word] = embedding
            except Exception as e:
                print(f"Failed to fetch embeddings for batch {i}. Reason: {str(e)}")
        self.save_embeddings_to_file(word_embeddings_map)
        return word_embeddings_map

    def closest_words(self, target_embedding, n=8):
        distances = [
            distance.euclidean(target_embedding, emb) for emb in self.embeddings
        ]
        sorted_indices = np.argsort(distances)
        return [self.words[idx] for idx in sorted_indices[:n]]

    def fetch_and_add_embeddings(self, words_to_embed, r):
        if not isinstance(words_to_embed, list):
            words_to_embed = [words_to_embed]

        words_to_fetch = []

        for word in words_to_embed:
            cached_embedding = self.redis_connection.get(f"word_embedding-{word}")
            if cached_embedding:
                print(f"Found cached embedding for {word}")
                self.word_embeddings_map[word] = pickle.loads(cached_embedding)
                if word not in self.words:
                    self.words.append(word)
                    self.embeddings = np.append(
                        self.embeddings, [pickle.loads(cached_embedding)], axis=0
                    )
            else:
                words_to_fetch.append(word)

        if not words_to_fetch:
            self.reduced_embeddings = self.pca.fit_transform(self.embeddings)
            return

        try:
            print(f"Fetching embeddings for {words_to_fetch}")
            response = openai.Embedding.create(input=words_to_fetch, model=self.model)
            new_embeddings = [word_data["embedding"] for word_data in response["data"]]
            for word, embedding in zip(words_to_fetch, new_embeddings):
                self.word_embeddings_map[word] = embedding
                if word not in self.words:
                    self.words.append(word)
                    self.embeddings = np.append(self.embeddings, [embedding], axis=0)
                # Store the new word and its embedding in Redis
                if word not in self.load_embeddings_from_file().keys():
                    r.set(f"word_embedding-{word}", pickle.dumps(embedding))
        except Exception as e:
            print(f"Failed to fetch embeddings. Reason: {str(e)}")
            print(f"Words causing the issue: {words_to_fetch}")
        self.reduced_embeddings = self.pca.fit_transform(self.embeddings)


class DashAppConfig:
    def __init__(self, manager):
        self.manager = manager
        self.app = Flask(__name__)
        self.dash_app = dash.Dash(
            __name__,
            server=self.app,
            routes_pathname_prefix="/",
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.dash_app.title = "LatentDictionary | Embeddings as a Dictionary"
        self.dash_app.index_string = index_string
        self.dash_app.layout = app_layout
        self.setup_callbacks()

    def setup_callbacks(self):
        r = redis.Redis(
            host="127.0.0.1",
            port=16379,
            password="56c443d446f145f0a52ade710646a19d",
            socket_timeout=60,
        )

        @self.dash_app.callback(
            [Output("closest-words", "children"), Output("word-input", "value")],
            [
                Input("word-input", "n_submit"),
                Input({"type": "word-tile", "index": ALL}, "n_clicks"),
            ],
            [
                State("word-input", "value"),
                State({"type": "word-tile", "index": ALL}, "children"),
            ],
        )
        def display_closest_words(
            word_submit_count, tile_clicks, word_input_value, tile_labels
        ):
            # Use manager's attributes instead of global variables
            words = self.manager.words
            word_embeddings_map = self.manager.word_embeddings_map

            words_to_highlight = []

            ctx = dash.callback_context
            component_id = None  # Initialize component_id with None

            if ctx.triggered:
                component_id = (
                    ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
                )

            if component_id and "word-tile" in component_id:
                clicked_button_idx = next(
                    (i for i, n in enumerate(tile_clicks) if n and n > 0), None
                )
                if clicked_button_idx is not None:
                    word_to_highlight = tile_labels[clicked_button_idx]
                else:
                    word_to_highlight = DEFAULT_WORD
            else:
                words_to_highlight = [
                    word.strip()
                    for word in (word_input_value or "").split(",")
                    if word.strip()
                ]
                word_to_highlight = (
                    words_to_highlight[0] if words_to_highlight else DEFAULT_WORD
                )

            # Fetch embeddings for words in the input if they aren't available
            words_not_in_map = [
                word for word in words_to_highlight if word not in word_embeddings_map
            ]
            if words_not_in_map:
                self.manager.fetch_and_add_embeddings(words_not_in_map, r)

            # Get the closest words
            if word_to_highlight not in self.manager.word_embeddings_map:
                return [
                    html.Div(f"Failed to fetch embedding for '{word_to_highlight}'.")
                ], dash.no_update

            target_embedding = self.manager.word_embeddings_map[word_to_highlight]
            closest_8 = self.manager.closest_words(target_embedding, 8)

            word_tiles = [
                dbc.Button(
                    word,
                    id={"type": "word-tile", "index": i},
                    className="mx-2 my-1 rounded-pill",
                    color="secondary",
                    outline=True,
                    n_clicks=0,
                )
                for i, word in enumerate(closest_8)
            ]

            if component_id is not None and "word-tile" in component_id:
                return [
                    html.Div(word_tiles)
                ], word_to_highlight  # update the input box with the clicked word
            else:
                return [
                    html.Div(word_tiles)
                ], dash.no_update  # no update for the input box

        @self.dash_app.callback(
            Output("3d-plot", "figure"),
            [
                Input("word-input", "n_submit"),
                Input({"type": "word-tile", "index": ALL}, "n_clicks"),
            ],
            [
                State("word-input", "value"),
                State({"type": "word-tile", "index": ALL}, "children"),
            ],
        )
        def update_graph(n_submit, tile_clicks, input_value, tile_labels):
            print("Updating graph")
            words = self.manager.words
            embeddings = self.manager.embeddings
            reduced_embeddings = self.manager.reduced_embeddings
            ctx = dash.callback_context

            clicked_button_idx = next(
                (i for i, n in enumerate(tile_clicks) if n and n > 0), None
            )

            # If a word-tile button was clicked, update the words_to_highlight
            if clicked_button_idx is not None:
                words_to_highlight = [tile_labels[clicked_button_idx]]
            else:
                words_to_highlight = [word.strip() for word in input_value.split(",")]

            colors = [
                "blue" if word not in words_to_highlight else "red" for word in words
            ]
            sizes = [20 if word in words_to_highlight else 5 for word in words]

            scatter = go.Scatter3d(
                x=reduced_embeddings[:, 0],
                y=reduced_embeddings[:, 1],
                z=reduced_embeddings[:, 2],
                mode="markers",
                marker=dict(color=colors, size=sizes),
                hovertext=words,
                hoverinfo="text",
            )

            layout = go.Layout(
                height=1000,
                margin=dict(l=0, r=0, b=0, t=0),
                uirevision="constant",
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
            )

            fig = go.Figure(data=[scatter], layout=layout)
            return fig

    def run(self):
        if __name__ == "__main__":
            self.app.run(host="0.0.0.0", debug=False, port=8050)


manager = EmbeddingsManager()
app_config = DashAppConfig(manager)
app = app_config.dash_app.server  # Flask server instance at the module level

if __name__ == "__main__":
    app_config.run()
