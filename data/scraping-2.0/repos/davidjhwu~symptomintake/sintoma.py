import dash
from dash import dcc, html, dash_table  # Updated import for dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import openai
import os
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
gpt_model = os.getenv("gpt_model")

# import pdfkit
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase
# from email import encoders
external_stylesheets = [
    'https://maxcdn.bootstrapcdn.com/bootswatch/4.5.2/journal/bootstrap.min.css',
    dbc.themes.BOOTSTRAP
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

style = {
    'padding': '3.5em',
    'backgroundColor': '#FFFFFF',  
    'fontFamily': 'Arial, sans-serif'
}

#Mapping for severity score of responses

severity_colors = {
    "Nenhuma": '#008000',
    "De maneira nenhuma": '#008000',
    "Nunca": '#008000',
    "Leve": '#90ee90',
    "Um pouco": '#90ee90',
    "Raramente": '#90ee90',
    "Moderada": '#ffff00',
    "Um tanto": '#ffff00',
    "Ocasionalmente": '#ffff00',
    "Severa": '#ffa500',
    "Bastante": '#ffa500',
    "Frequentemente": '#ffa500',
    "Muito severa": '#ff0000',
    "Muito": '#ff0000',
    "Quase constantemente": '#ff0000',
    "Não": '#008000',
    "Sim": '#ff0000',
}
def create_data_table():
    style_data_conditional = []

    for response, color in severity_colors.items():
        text_color = 'white' if color != '#ffff00' else 'black'
        style_data_conditional.append({
            'if': {
                'filter_query': '{{answer}} = "{}"'.format(response),
                'column_id': 'answer'
            },
            'backgroundColor': color,
            'color': text_color
        })

    return dash_table.DataTable(
        id='results_table',
        columns=[
            {'name': 'Pergunta', 'id': 'question'},
            {'name': 'Resposta', 'id': 'answer'},
        ],
        data=[],
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
            'textAlign': 'center',
            'border': 'none',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': 'none',
        },
        style_table={
            'margin': '0 auto',
            'width': '50%'
        },
        style_data_conditional=style_data_conditional
    )

app.layout = html.Div([

html.Img(src=app.get_asset_url('inca.png'), style={'height': '100px', 'width': 'auto'}),
    # ... the rest of your layout
dcc.Markdown('# Formulário de Revisão de Sintomas do Paciente em Radioterapia'),
html.P([html.Br()]),
dcc.Markdown('#### Por favor, responda às seguintes perguntas sobre seus sintomas atuais'),
dcc.Markdown('Cada formulário deve ser preenchido com cuidado, os resultados serão enviados ao seu médico'),
dcc.Markdown('#### Perguntas Gerais'),
dcc.Markdown('###### Quantos tratamentos de radiação você já fez? Tudo bem se você não souber.'),
dcc.Input(
    id='number_of_treatments',
    placeholder='Insira um valor',
    type='text',
    value='ex: 3, ou Não sei'),
html.P([html.Br()]),
dcc.Markdown('### Perguntas sobre Sintomas'),
dcc.Markdown('Para cada uma das seguintes perguntas, vou pedir que você classifique seus sintomas.'),
    dcc.Markdown("#### Fadiga"),
dcc.Markdown(
    "###### Nos últimos 7 dias, qual foi a GRAVIDADE da sua FADIGA, CANSAÇO OU FALTA DE ENERGIA no seu pior momento?"
),
dcc.Dropdown(
    id="fatigue_severity",
    options=[
        {"label": "Nenhuma", "value": "Nenhuma"},
        {"label": "Leve", "value": "Leve"},
        {"label": "Moderada", "value": "Moderada"},
        {"label": "Severa", "value": "Severa"},
        {"label": "Muito severa", "value": "Muito severa"},
    ],
    value=None,
),
dcc.Markdown(
    "###### Nos últimos 7 dias, quanto a FADIGA, CANSAÇO OU FALTA DE ENERGIA interferiram nas suas atividades habituais ou diárias?"
),
dcc.Dropdown(
    id="fatigue_interference",
    options=[
        {"label": "De maneira nenhuma", "value": "De maneira nenhuma"},
        {"label": "Um pouco", "value": "Um pouco"},
        {"label": "Um tanto", "value": "Um tanto"},
        {"label": "Bastante", "value": "Bastante"},
    ],
    value=None,
),
html.P([html.Br()]),
dcc.Markdown('#### Gases'),
dcc.Markdown('###### Nos últimos 7 dias, você teve um AUMENTO na EMISSÃO DE GASES (FLATULÊNCIA)?'),
dcc.Dropdown(
    id='gas',
    options=[
        {'label': 'Sim', 'value': 'Sim'},
        {'label': 'Não', 'value': 'Não'}
    ],
    value=None,
),
dcc.Markdown('#### Diarreia'),      
dcc.Markdown('###### Nos últimos 7 dias, com que FREQUÊNCIA você teve FEZES SOLTAS OU LÍQUIDAS (DIARREIA)?'),
dcc.Dropdown(
    id='diarrhea_frequency',
    options=[
        {'label': 'Nunca', 'value': 'Nunca'},
        {'label': 'Raramente', 'value': 'Raramente'},
        {'label': 'Ocasionalmente', 'value': 'Ocasionalmente'},
        {'label': 'Frequentemente', 'value': 'Frequentemente'},
        {'label': 'Quase constantemente', 'value': 'Quase constantemente'}
    ],
    value=None,
),
dcc.Markdown('#### Dor Abdominal'),
dcc.Markdown('###### Nos últimos 7 dias, com que FREQUÊNCIA você sentiu DOR NO ABDÔMEN (REGIÃO DA BARRIGA)?'),
dcc.Dropdown(
    id='abdominal_pain_frequency',
    options=[
        {'label': 'Nunca', 'value': 'Nunca'},
        {'label': 'Raramente', 'value': 'Raramente'},
        {'label': 'Ocasionalmente', 'value': 'Ocasionalmente'},
        {'label': 'Frequentemente', 'value': 'Frequentemente'},
        {'label': 'Quase constantemente', 'value': 'Quase constantemente'}
    ],
    value=None,
),

dcc.Markdown('###### Nos últimos 7 dias, qual foi a GRAVIDADE da sua DOR NO ABDÔMEN (REGIÃO DA BARRIGA) no seu pior momento?'),
dcc.Dropdown(
    id='abdominal_pain_severity',
    options=[
        {'label': 'Nenhuma', 'value': 'Nenhuma'},
        {'label': 'Leve', 'value': 'Leve'},
        {'label': 'Moderada', 'value': 'Moderada'},
        {'label': 'Severa', 'value': 'Severa'},
        {'label': 'Muito severa', 'value': 'Muito severa'}
    ],
    value=None,
),

dcc.Markdown('###### Nos últimos 7 dias, quanto a DOR NO ABDÔMEN (REGIÃO DA BARRIGA) interferiu nas suas atividades habituais ou diárias?'),
dcc.Dropdown(
    id='abdominal_pain_adl',
    options=[
        {'label': 'De maneira nenhuma', 'value': 'De maneira nenhuma'},
        {'label': 'Um pouco', 'value': 'Um pouco'},
        {'label': 'Um tanto', 'value': 'Um tanto'},
        {'label': 'Bastante', 'value': 'Bastante'},
        {'label': 'Muito', 'value': 'Muito'}
    ],
    value=None,
    ),
html.P([html.Br()]),
 dcc.Markdown('Agora vamos discutir seus sintomas urinários.'),
dcc.Markdown('#### Sintomas Urinários'),
dcc.Markdown('##### Micção Dolorosa'),
dcc.Markdown('###### Nos últimos 7 dias, qual foi a GRAVIDADE da sua DOR OU QUEIMAÇÃO AO URINAR no seu pior momento?'),
dcc.Dropdown(
    id='painful_urination_severity',
    options=[
        {'label': 'Nenhuma', 'value': 'Nenhuma'},
        {'label': 'Leve', 'value': 'Leve'},
        {'label': 'Moderada', 'value': 'Moderada'},
        {'label': 'Severa', 'value': 'Severa'},
        {'label': 'Muito severa', 'value': 'Muito severa'}
    ],
    value=None,
),
dcc.Markdown('##### Urgência Urinária'),
dcc.Markdown('###### Nos últimos 7 dias, com que FREQUÊNCIA você sentiu uma URGENCIA REPENTINA DE URINAR?'),
dcc.Dropdown(
    id='urinary_urgency_frequency',
    options=[
        {'label': 'Nunca', 'value': 'Nunca'},
        {'label': 'Raramente', 'value': 'Raramente'},
        {'label': 'Ocasionalmente', 'value': 'Ocasionalmente'},
        {'label': 'Frequentemente', 'value': 'Frequentemente'},
        {'label': 'Quase constantemente', 'value': 'Quase constantemente'}
    ],
    value=None,
),
dcc.Markdown('###### Nos últimos 7 dias, quanto as REPENTINAS URGENCIAS DE URINAR interferiram nas suas atividades habituais ou diárias?'),
dcc.Dropdown(
    id='urinary_urgency_adl',
    options=[
        {'label': 'De maneira nenhuma', 'value': 'De maneira nenhuma'},
        {'label': 'Um pouco', 'value': 'Um pouco'},
        {'label': 'Um tanto', 'value': 'Um tanto'},
        {'label': 'Bastante', 'value': 'Bastante'},
        {'label': 'Muito', 'value': 'Muito'}
    ],
    value=None,
),
dcc.Markdown('##### Frequência Urinária'),
dcc.Markdown('###### Nos últimos 7 dias, houve momentos em que você teve que URINAR FREQUENTEMENTE?'),
dcc.Dropdown(
    id='urinary_frequency',
    options=[
        {'label': 'Nunca', 'value': 'Nunca'},
        {'label': 'Raramente', 'value': 'Raramente'},
        {'label': 'Ocasionalmente', 'value': 'Ocasionalmente'},
        {'label': 'Frequentemente', 'value': 'Frequentemente'},
        {'label': 'Quase constantemente', 'value': 'Quase constantemente'}
    ],
    value=None,
),
dcc.Markdown('###### Nos últimos 7 dias, quanto a FREQUENTE MICÇÃO interferiu nas suas atividades habituais ou diárias?'),
dcc.Dropdown(
    id='urinary_frequency_interference',
    options=[
        {'label': 'De maneira nenhuma', 'value': 'De maneira nenhuma'},
        {'label': 'Um pouco', 'value': 'Um pouco'},
        {'label': 'Um tanto', 'value': 'Um tanto'},
        {'label': 'Bastante', 'value': 'Bastante'},
        {'label': 'Muito', 'value': 'Muito'}
    ],
    value=None,
),
dcc.Markdown('##### Mudança na Cor Usual da Urina'),
dcc.Markdown('###### Nos últimos 7 dias, você teve alguma MUDANÇA NA COR DA URINA?'),
dcc.Dropdown(
    id='urine_color_change',
    options=[
        {'label': 'Sim', 'value': 'Sim'},
        {'label': 'Não', 'value': 'Não'}
    ],
    value=None,
),
dcc.Markdown('##### Incontinência Urinária'),
dcc.Markdown('###### Nos últimos 7 dias, com que FREQUÊNCIA você teve PERDA DE CONTROLE DA URINA (VAZAMENTO)?'),
dcc.Dropdown(
    id='urinary_incontinence_frequency',
    options=[
        {'label': 'Nunca', 'value': 'Nunca'},
        {'label': 'Raramente', 'value': 'Raramente'},
        {'label': 'Ocasionalmente', 'value': 'Ocasionalmente'},
        {'label': 'Frequentemente', 'value': 'Frequentemente'},
        {'label': 'Muito', 'value': 'Muito'},
        {'label': 'Quase constantemente', 'value': 'Quase constantemente'}
    ],
    value=None,
),
dcc.Markdown('###### Nos últimos 7 dias, quanto a PERDA DE CONTROLE DA URINA (VAZAMENTO) interferiu nas suas atividades habituais ou diárias?'),
dcc.Dropdown(
    id='urinary_incontinence_interference',
    options=[
        {'label': 'De maneira nenhuma', 'value': 'De maneira nenhuma'},
        {'label': 'Um pouco', 'value': 'Um pouco'},
        {'label': 'Um tanto', 'value': 'Um tanto'},
        {'label': 'Bastante', 'value': 'Bastante'},
        {'label': 'Muito', 'value': 'Muito'}
    ],
    value=None,
),
html.P([html.Br()]),
dcc.Markdown("#### Reação da Pele à Radiação"),
dcc.Markdown(
    "###### Nos últimos 7 dias, qual foi a GRAVIDADE das QUEIMADURAS NA PELE DEVIDO À RADIAÇÃO no seu pior momento?"
),
dcc.Dropdown(
    id="radiation_skin_reaction_severity",
    options=[
        {"label": "Nenhuma", "value": "Nenhuma"},
        {"label": "Leve", "value": "Leve"},
        {"label": "Moderada", "value": "Moderada"},
        {"label": "Severa", "value": "Severa"},
        {"label": "Muito severa", "value": "Muito severa"},
        {"label": "Não aplicável", "value": "Não aplicável"},
    ],
    value=None,
),
html.P([html.Br()]),
dcc.Markdown('#### Última Pergunta!'),
dcc.Markdown('###### Finalmente, você tem outros sintomas que deseja relatar?'),
dcc.Input(
    id='additional_symptoms',
    placeholder='Digite aqui...',
    type='text',
    value=''),
html.P([html.Br()]),

html.Div(className="d-grid gap-2 d-flex justify-content-center", children=[
    dcc.Loading(id="loading", type="circle", children=[
        html.Button("Enviar", id="submit_button", n_clicks=0, className="btn btn-lg btn-primary", style={"width": "200px"})
    ]),
]),
html.Br(),
html.Div([
    html.Div([
        html.Div('GPT 4', className='card-header'),
        dcc.Loading(id="loading-summary", type="circle", children=[
            html.Div([
                html.H4('Resumo dos Sintomas do Paciente em Oncologia Radiológica', className='card-title'),
                html.P(id='summary', className='card-text')
            ], className='card-body')
        ])
    ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
], className='summary-container mx-auto', style={'width': '60%'}),
html.Br(),
html.Div([
    dcc.Markdown('### Resultados da Pesquisa')
], style={'textAlign': 'center'}),
create_data_table(),
html.Br(),
dcc.Markdown("""#### Aviso Legal"""),
'Esta aplicação não armazena nenhum dos dados fornecidos. No entanto, como está utilizando a API OpenAI GPT, a OpenAI poderia potencialmente usar as informações fornecidas. Não forneça nenhuma informação pessoal.',
html.Br(),
html.Br(),
'Todas as informações contidas nesta aplicação são apenas para fins informativos gerais e não constituem aconselhamento médico ou profissional. As informações encontradas nesta aplicação são fornecidas como estão e na medida permitida por lei, os criadores renunciam a todas as outras garantias de qualquer tipo, sejam expressas, implícitas, estatutárias ou de outra forma. Em nenhum caso os autores ou seus afiliados serão responsáveis por quaisquer danos de qualquer tipo, incluindo, sem limitação, danos diretos, indiretos, incidentais, especiais, exemplares, punitivos ou consequentes, decorrentes do uso, incapacidade de usar ou dos resultados do uso desta aplicação, de quaisquer sites vinculados a esta aplicação ou dos materiais, informações ou serviços contidos em todos esses sites, seja com base em garantia, contrato, delito ou qualquer outra teoria legal e independentemente de ter sido ou não avisado da possibilidade de tais danos. Se o uso dos materiais, informações ou serviços desta aplicação ou de qualquer site vinculado a esta aplicação resultar na necessidade de manutenção, reparo ou correção de equipamentos ou dados, você assume todos os custos relacionados.)',
html.Br(),
html.Br(),
dcc.Markdown("""#### Sobre"""),
'Criado por David JH Wu & Jean-Emmanuel Bibault - 2023',
], style=style)

@app.callback(
    Output('summary', 'children'),
    Output('results_table', 'data'),
    Input('submit_button', 'n_clicks'),
    State('number_of_treatments', 'value'),
    State('gas', 'value'),
    State('diarrhea_frequency', 'value'),
    State('abdominal_pain_frequency', 'value'),
    State('abdominal_pain_severity', 'value'),
    State('abdominal_pain_adl', 'value'),
    State('painful_urination_severity', 'value'),
    State('urinary_urgency_frequency', 'value'),
    State('urinary_urgency_adl', 'value'),
    State('urinary_frequency', 'value'),
    State('urinary_frequency_interference', 'value'),
    State('urine_color_change', 'value'),
    State('urinary_incontinence_frequency', 'value'),
    State('urinary_incontinence_interference', 'value'),
    State('radiation_skin_reaction_severity', 'value'),
    State('fatigue_severity', 'value'),
    State('fatigue_interference', 'value'),
    State('additional_symptoms', 'value'),
)
def update_table_results(n_clicks, *responses):
    if n_clicks == 0:
        return None, []

    questions = [
        'Número de tratamentos de radiação',
        'Aumento na emissão de gases',
        'Frequência de diarreia',
        'Frequência de dor abdominal',
        'Gravidade da dor abdominal',
        'Dor abdominal com Atividades da Vida Diária',
        'Gravidade da micção dolorosa',
        'Frequência de urgência urinária',
        'Urgência urinária com Atividades da Vida Diária',
        'Frequência urinária',
        'Frequência urinária com Atividades da Vida Diária',
        'Mudança na cor da urina',
        'Frequência de incontinência urinária',
        'Incontinência urinária com Atividades da Vida Diária',
        'Gravidade da reação da pele à radiação',
        'Gravidade da fadiga',
        'Fadiga com Atividades da Vida Diária',
        'Sintomas adicionais',
    ]

    data = [{'question': question, 'answer': response} for question, response in zip(questions, responses)]
    summary = summarize_table(data)
    return summary, data

def summarize_table(data):
    messages = [{
        'role': 'system',
        'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please take a deep breath, think step-by-step, and summarize the following data into three sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the Brazilian Portuguese language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
    }]
    
    for row in data:
        messages.append({
            'role': 'user',
            'content': f"{row['question']}: {row['answer']}"
        })
    
    messages.append({
        'role': 'assistant',
        'content': "Summary:"
    })

    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        n=1,
        stop=None,
        temperature=0.2,
    )

    summary = response.choices[0].message.content.strip()
    return summary

if __name__ == '__main__':
    app.run_server(debug=True)
