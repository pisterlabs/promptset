from lib.app_runner import run_app

run_app("cs514", 8050)

# import dash
# from dash import dcc
# from dash import html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State

# import openai
# import numpy as np
# import os

# OPENAI_KEY_PATH = "openai_key.txt"

# DATA_DIR = "data"
# NAME = "cs514"
# save_dir = os.path.join(DATA_DIR, NAME)

# embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
# links = np.load(os.path.join(save_dir, "links.npy"))
# documents = np.load(os.path.join(save_dir, "documents.npy"))
# img_files = np.load(os.path.join(save_dir, "img_files.npy"))

# with open(OPENAI_KEY_PATH, "r") as f:
#     openai.api_key = f.read().strip()

# def get_embeddings(texts, type, model="ada"):
#     results = openai.Embedding.create(input=texts, engine=f"text-search-{model}-{type}-001")['data']
#     return np.array(list(map(lambda x: x['embedding'], results)))

# save_dir = os.path.join(DATA_DIR, NAME)


# debug = True
# theme = "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/materia/bootstrap.min.css"
# app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[theme], assets_folder=os.path.join(save_dir, "imgs"))

# app.layout = html.Div([
#     dcc.Store(id="session", storage_type='local'),
#     dcc.Location(id='url', refresh=False),
#     # Navbar
#     dbc.NavbarSimple(id="navbar", brand="CS 685 Semantic Search", brand_href="/todo", color="primary", className="mb-3", dark=True),
#     html.Meta(name="viewport", content="width=device-width, initial-scale=1"),
    
#     # search bar
#     dbc.Row([
#         dbc.Col(),
#         dbc.Col(
#             dbc.Input(id="search-input", type="text", placeholder="Search for a question", debounce=True, className="mb-3", maxLength=500),
#             width=4,
#         ),
#         dbc.Col(
#             dbc.Button("Search", id="search-button", color="primary", className="mb-3"),
#             width=1
#         ),
#         dbc.Col(),
#     ]),

#     # results - headings with associated png images
#     dbc.Row([
#         dbc.Col(),
#         dbc.Col([
#             dcc.Loading(id="search-results"),
#             # button to show more results
#             # dbc.Button("Show more results", id="show-more-results", color="primary", className="mb-3", style={"display": "none"}),
#         ], width=5, style={"padding-top": "20px"}),
#         dbc.Col(),
#     ])
# ])


# @app.callback(
#     Output("search-results", "children"),
#     Input("search-button", "n_clicks"),
#     Input("search-input", "value")
# )
# def search_results(n_clicks, query):
#     if query is None:
#         return dash.no_update
    
#     top_k = 50

#     query_embedding = get_embeddings([query], "query", model="curie")[0]
#     # store the query embedding in the session
#     # session = dcc.Store.get("session")
#     # session["query_embedding"] = query_embedding
#     # dcc.Store.set("session", session)

#     # Find the embedding that maximizes the cosine similarity
#     similarity = np.dot(embeddings, query_embedding)

#     # Find the index of the top k most similar embeddings
#     top_k_indices = np.argsort(similarity)[-top_k:]
#     top_k_indices = top_k_indices[::-1]

#     children = []

#     for i in top_k_indices:
#         link = links[i] # url to the webpage
#         img_file = img_files[i] # path to the image
#         # img_file_path = os.path.join(save_dir, "imgs", img_file) # path to the image

#         # example link https://people.cs.umass.edu/~mcgregor/514S22/lecture1.pdf#page=51
#         lecture_number = link.split("/")[-1].split(".")[0].split("lecture")[-1]
#         page_number = link.split("#page=")[-1]
#         heading_text = "Lecture %s, slide %s - %d%% match" % (lecture_number, page_number, round(similarity[i] * 100))

#         children.append(
#             html.Div([
#                 # make the link clickable
#                 # and open in new tab
#                 html.A(
#                     # the text of the link
#                     html.H5(heading_text),
#                     # the url
#                     href=link,
#                     # open in new tab
#                     target="_blank",
#                 ),
#                 # make the image larger
#                 html.Img(src=app.get_asset_url(img_file), style={"width": "70%"}),
#             ])
#         )
    
#     return children



# if __name__ == '__main__':
#     app.run_server(debug=debug, host="0.0.0.0", port=8051)