from utils import utils, utils_tests, utils_google, utils_chatgpt
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import openai
import json

# Función para validar el EMAIL del usuario
def validate_email(email_input, email_selected):
    if email_input.lower() == email_selected.lower():
        return True
    else:
        return False

# Función para generar el reto técnico
def generate_technical_challenge(position, level, example, user_input):
    messages = [
        {"role": "system", "content": "Eres un generador de retos técnicos cortos y creativos de tecnología en español. Los niveles de seniority son Junior, Medium, SemiSenior y Senior. Cada nivel tiene una mayor expectativa de habilidades y experiencia."},
        {"role": "user", "content": f"Necesito un reto técnico para un postulante al puesto {position} de nivel {level}, da las indicaciones específicas para este nivel. El reto debe estar estimado para 6 horas máximo, no des retos muy largos, sé creativo. Por favor escríbelo tomando en cuenta que lo leerá el candidato final. Este es un ejemplo de reto técnico, solo un ejemplo, sé creativo pero con el mismo estilo y testeando mismas habilidades: {example}. Mi experiencia y perfil es el siguiente: {user_input}."},
    ]
    print(messages[1]["content"])
    response = utils_chatgpt.chat_chatgpt(messages)
    return response

def serve_layout():
    layout = html.Div(
        [
            # Título
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(
                            "Postulaciones",
                            id="title-tests",
                            className="text-center",
                            style={"color":"#1b2b58", "font-size":"28px"}
                        ),
                        sm=12,
                        lg=12
                    ),
                    dbc.Col(
                        html.H2(
                            "Pruebas Técnicas",
                            id="subtitle-tests",
                            className="text-center",
                            style={"color":"#1b2b58", "font-size":"21px", "margin-bottom": "20px"}
                        ),
                        sm=12,
                        lg=12
                    ),
                ],
                justify='center',
            ),

            # Dropdown para seleccionar el nombre del usuario y Input para ingresar el EMAIL
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                html.Strong("1. Elige tu nombre"),
                                className="label"
                            ),
                            dcc.Dropdown(id='dropdown-user-tests'),
                            html.Label(
                                html.Strong("2. Ingresa tu email"),
                                className="label"
                            ),
                            dcc.Input(id='input-email-tests', type='text'),
                            dcc.Loading(
                                id="loading-validate",
                                type="circle",
                                children=html.Button('Validar', id='btn-validate-tests')
                            ),
                            html.Div(id='div-error-tests'),
                        ],
                        sm=12,
                        md=6,
                        lg=4
                    ),
                ]
            ),

            # Dropdown para seleccionar el puesto abierto y el nivel
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                html.Strong("3. Elige el puesto al que postulas"),
                                className="label"
                            ),
                            dcc.Dropdown(id='dropdown-position-tests', style={'display': 'none'}),
                            html.Label(
                                html.Strong("4. Elige tu nivel"),
                                className="label"
                            ),
                            dcc.Dropdown(
                                id='dropdown-level-tests',
                                options=[
                                    {'label': 'Junior', 'value': 'Junior'},
                                    {'label': 'Medium', 'value': 'Medium'},
                                    {'label': 'SemiSenior', 'value': 'SemiSenior'},
                                    {'label': 'Senior', 'value': 'Senior'}
                                ],
                                style={'display': 'none'}
                            ),
                            dcc.Loading(
                                id="loading-generate",
                                type="circle",
                                children=html.Button('Generar reto', id='btn-generate-tests', style={'display': 'none'})
                            ),
                        ],
                        sm=12,
                        md=6,
                        lg=4
                    ),
                ]
            ),

            # Reto técnico generado
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id='div-challenge-tests', style={'display': 'none', 'border': '1px solid #ddd', 'padding': '20px', 'border-radius': '8px', 'background-color': '#f9f9f9', 'text-align': 'center'}),
                        sm=12,
                        md=12,
                        lg=12
                    ),
                ],
                style={'margin-top': '20px'}
            ),

            # Botón para resolver el reto
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-resolve",
                            type="circle",
                            children=html.Button('Deseo resolver el reto', id='btn-resolve-tests', style={'display': 'none'})
                        ),
                        sm=12,
                        md=6,
                        lg=4
                    ),
                ],
                style={'margin-top': '20px'}
            ),

            # Mensaje después de resolver el reto
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id='div-resolve-message', style={'display': 'none'}),
                        sm=12,
                        md=12,
                        lg=12
                    ),
                ],
                style={'margin-top': '20px'}
            ),
        ],
        className="dash-inside-container",
    )
    return layout

def init_callbacks(dash_app):
    @dash_app.callback(
        Output('dropdown-user-tests', 'options'),
        Input('dropdown-user-tests', 'id')
    )
    def update_user_dropdown(id):
        # Leer los nombres de los usuarios de Google Sheets
        ws = utils_google.open_ws('Postula a Digitalia (Responses)', 'Form Responses 1')
        df = utils_google.read_ws_data(ws)
        df["names"] = df["Nombres"] + " " + df["Apellidos"]
        options = [{'label': name, 'value': mail} for name, mail in zip(df['names'], df['Email Address'])]
        return options

    @dash_app.callback(
        [Output('div-error-tests', 'children'),
         Output('dropdown-position-tests', 'style'),
         Output('dropdown-level-tests', 'style'),
         Output('btn-generate-tests', 'style')],
        Input('btn-validate-tests', 'n_clicks'),
        State('input-email-tests', 'value'),
        State('dropdown-user-tests', 'value')
    )
    def validate_user(n_clicks, email_input, email_selected):
        if n_clicks is not None:
            if not validate_email(email_input, email_selected):
                return "Usuario no encontrado, por favor vuelva a intentar", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
            else:
                return "", {}, {}, {}

    @dash_app.callback(
        Output('dropdown-position-tests', 'options'),
        Input('dropdown-position-tests', 'id')
    )
    def update_position_dropdown(id):
        #Leer los puestos abiertos de Google Sheets
        ws = utils_google.open_ws('Postula a Digitalia (Responses)', 'portal-positions')
        df = utils_google.read_ws_data(ws)
        options = [{'label': position, 'value': position} for position in df['position']]
        return options

    @dash_app.callback(
        [Output('div-challenge-tests', 'children'),
         Output('div-challenge-tests', 'style'),
         Output('btn-resolve-tests', 'style')],
        Input('btn-generate-tests', 'n_clicks'),
        State('dropdown-position-tests', 'value'),
        State('dropdown-level-tests', 'value'),
        State('dropdown-user-tests', 'value')
    )
    def update_technical_challenge(n_clicks, position, level, user_email):
        if n_clicks is not None and level is not None and position is not None:
            # Leer el ejemplo de Google Sheets
            ws = utils_google.open_ws('Postula a Digitalia (Responses)', 'portal-positions')
            df = utils_google.read_ws_data(ws)
            example = df.loc[df['position'] == position, 'example'].iloc[0]

            # Leer la información del usuario
            ws_user = utils_google.open_ws('Postula a Digitalia (Responses)', 'Form Responses 1')
            df_user = utils_google.read_ws_data(ws_user)
            user_input = df_user.loc[df_user['Email Address'] == user_email, 'Cuéntanos sobre ti cuando trabajas, skills, forma de ser, etc.'].iloc[0]

            # Generar el reto técnico
            challenge = generate_technical_challenge(position, level, example, user_input)
            return challenge, {}, {}
        return "", {'display': 'none'}, {'display': 'none'}

    @dash_app.callback(
        [Output('div-resolve-message', 'children'),
         Output('div-resolve-message', 'style')],
        Input('btn-resolve-tests', 'n_clicks'),
        State('dropdown-user-tests', 'value'),
        State('dropdown-position-tests', 'value'),
        State('dropdown-level-tests', 'value'),
        State('div-challenge-tests', 'children')
    )
    def resolve_challenge(n_clicks, user, position, level, challenge):
        if n_clicks is not None:
            # Leer los retos de Google Sheets
            ws = utils_google.open_ws('Postula a Digitalia (Responses)', 'portal-retos')
            df = utils_google.read_ws_data(ws)

            # Añadir el nuevo reto
            new_row = {'User': user, 'Position': position, 'Level': level, 'Challenge': challenge}
            df = df.append(new_row, ignore_index=True)

            # Guardar los retos en Google Sheets
            utils_google.pandas_to_sheets(df, ws)

            return "Genial, procede a desarrollar el reto, nuestro equipo de hunting te contactará pronto", {}
        return None, {'display': 'none'}
