import dash
from dash import dcc, html
import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import openai
import os
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
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
    "None": '#008000',
    "Not at all": '#008000',
    "Never": '#008000',
    "Mild": '#90ee90',
    "A little bit": '#90ee90',
    "Rarely": '#90ee90',
    "Moderate": '#ffff00',
    "Somewhat": '#ffff00',
    "Occasionally": '#ffff00',
    "Severe": '#ffa500',
    "Quite a bit": '#ffa500',
    "Frequently": '#ffa500',
    "Very severe": '#ff0000',
    "Very much": '#ff0000',
    "Almost constantly": '#ff0000',
    "No": '#008000',
    "Yes": '#ff0000',
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
            {'name': 'Question', 'id': 'question'},
            {'name': 'Answer', 'id': 'answer'},
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
dcc.Markdown('# Prostate Radiotherapy Patient Symptom Intake Form'),
  html.P([html.Br()]),
  dcc.Markdown('#### Please answer the following questions about your current symptoms'),
  dcc.Markdown('Each form must be carefully filled out, results will be sent to your physician'),
  dcc.Markdown('#### General Questions'),
  dcc.Markdown('###### How many radiation treatments have you had? It\'s okay if you don\'t know.'),
  dcc.Input(
      id='number_of_treatments',
      placeholder='Enter a value',
      type='text',
      value='ie 3, or I don\'t know'),
  html.P([html.Br()]),
  dcc.Markdown('### Symptom Questions'),
  dcc.Markdown('For each of the following question I\'m going to ask you to grade your symptoms.'),
      dcc.Markdown("#### Fatigue"),
    dcc.Markdown(
        "###### In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST?"
    ),
    dcc.Dropdown(
        id="fatigue_severity",
        options=[
            {"label": "None", "value": "None"},
            {"label": "Mild", "value": "Mild"},
            {"label": "Moderate", "value": "Moderate"},
            {"label": "Severe", "value": "Severe"},
            {"label": "Very severe", "value": "Very severe"},
        ],
        value=None,
    ),

    dcc.Markdown(
        "###### In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities?"
    ),
    dcc.Dropdown(
        id="fatigue_interference",
        options=[
            {"label": "Not at all", "value": "Not at all"},
            {"label": "A little bit", "value": "A little bit"},
            {"label": "Somewhat", "value": "Somewhat"},
            {"label": "Quite a bit", "value": "Quite a bit"},
        ],
        value=None,
    ),
    html.P([html.Br()]),
  dcc.Markdown('#### Gas'),
    dcc.Markdown('###### In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)?'),
    dcc.Dropdown(
        id='gas',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'}
        ],
        value=None,
    ),
  dcc.Markdown('#### Diarrhea'),      
  dcc.Markdown('###### In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)?'),
    dcc.Dropdown(
    id='diarrhea_frequency',
    options=[
        {'label': 'Never', 'value': 'Never'},
        {'label': 'Rarely', 'value': 'Rarely'},
        {'label': 'Occasionally', 'value': 'Occasionally'},
        {'label': 'Frequently', 'value': 'Frequently'},
        {'label': 'Almost constantly', 'value': 'Almost constantly'}
    ],
    value=None,
    ),

    dcc.Markdown('#### Abdominal Pain'),
    dcc.Markdown('###### In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)?'),
    dcc.Dropdown(
        id='abdominal_pain_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Almost constantly', 'value': 'Almost Constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST?'),
    dcc.Dropdown(
        id='abdominal_pain_severity',
        options=[
            {'label': 'None', 'value': 'None'},
            {'label': 'Mild', 'value': 'Mild'},
            {'label': 'Moderate', 'value': 'Moderate'},
            {'label': 'Severe', 'value': 'Severe'},
            {'label': 'Very severe', 'value': 'Very severe'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='abdominal_pain_adl',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),
  html.P([html.Br()]),
  dcc.Markdown('Now let\'s discuss your urinary symptoms.'),
  dcc.Markdown('#### Urinary Symptoms'),
    dcc.Markdown('##### Painful Urination'),
    dcc.Markdown('###### In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST?'),
    dcc.Dropdown(
        id='painful_urination_severity',
        options=[
            {'label': 'None', 'value': 'None'},
            {'label': 'Mild', 'value': 'Mild'},
            {'label': 'Moderate', 'value': 'Moderate'},
            {'label': 'Severe', 'value': 'Severe'},
            {'label': 'Very severe', 'value': 'Very severe'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Urinary Urgency'),
    dcc.Markdown('###### In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN?'),
    dcc.Dropdown(
        id='urinary_urgency_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Almost constantly', 'value': 'Almost constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='urinary_urgency_adl',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Urinary Frequency'),
    dcc.Markdown('###### In the last 7 days, were there times when you had to URINATE FREQUENTLY?'),
    dcc.Dropdown(
        id='urinary_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Almost constantly', 'value': 'Almost constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='urinary_frequency_interference',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Change in Usual Urine Color'),
    dcc.Markdown('###### In the last 7 days, did you have any URINE COLOR CHANGE?'),
    dcc.Dropdown(
        id='urine_color_change',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Urinary Incontinence'),
    dcc.Markdown('###### In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)?'),
    dcc.Dropdown(
        id='urinary_incontinence_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Very much', 'value': 'Very much'},
            {'label': 'Almost constantly', 'value': 'Almost constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='urinary_incontinence_interference',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),
    html.P([html.Br()]),
    dcc.Markdown("#### Radiation Skin Reaction"),
    dcc.Markdown(
        "###### In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST?"
    ),
    dcc.Dropdown(
        id="radiation_skin_reaction_severity",
        options=[
            {"label": "None", "value": "None"},
            {"label": "Mild", "value": "Mild"},
            {"label": "Moderate", "value": "Moderate"},
            {"label": "Severe", "value": "Severe"},
            {"label": "Very severe", "value": "Very severe"},
            {"label": "Not applicable", "value": "Not applicable"},
        ],
        value=None,
    ),
    html.P([html.Br()]),
    dcc.Markdown('#### Last Question!'),
    dcc.Markdown('###### Finally, do you have any other symptoms that you wish to report?'),
    dcc.Input(
        id='additional_symptoms',
        placeholder='Type here...',
        type='text',
        value=''),
    html.P([html.Br()]),
    dcc.Markdown(
        "###### Summarization Language"
    ),
    dcc.Dropdown(
        id="language",
        options=[
            {"label": "English", "value": "English"},
            {"label": "French", "value": "French"},
            {"label": "Portuguese", "value": "Portuguese"},
        ],
        value=None,
    ),
    html.Div(className="d-grid gap-2 d-flex justify-content-center", children=[
        dcc.Loading(id="loading", type="circle", children=[
            html.Button("Submit", id="submit_button", n_clicks=0, className="btn btn-lg btn-primary", style={"width": "200px"})
        ]),
    ]),
    html.Br(),
    html.Div([
        html.Div([
            html.Div('GPT 4', className='card-header'),
            dcc.Loading(id="loading-summary", type="circle", children=[
                html.Div([
                    html.H4('Radiation Oncology Patient Symptom Summary', className='card-title'),
                    html.P(id='summary', className='card-text')
                ], className='card-body')
            ])
        ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
    ], className='summary-container mx-auto', style={'width': '60%'}),
    html.Br(),
    html.Div([
        dcc.Markdown('### Survey Results')
    ], style={'textAlign': 'center'}),
    create_data_table(),
    html.Br(),
    dcc.Markdown("""#### Disclaimer"""),
    'This application does not store any of the data provided. However, as it is using the ChatGPT API, OpenAI could potentially use the information provided. Do not provide any personal information.',
    html.Br(),
    html.Br(),
    'All information contained on this application is for general informational purposes only and does not constitute medical or other professional advice. It should not be used for medical purpose. The information you find on this application is provided as is and to the extent permitted by law, the creators disclaim all other warranties of any kind, whether express, implied, statutory or otherwise. In no event will the authors or its affiliates be liable for any damages whatsoever, including without limitation, direct, indirect, incidental, special, exemplary, punitive or consequential damages, arising out of the use, inability to use or the results of use of this application, any websites linked to this application or the materials or information or services contained at any or all such websites, whether based on warranty, contract, tort or any other legal theory and whether or not advised of the possibility of such damages. If your use of the materials, information or services from this application or any website linked to this application results in the need for servicing, repair or correction of equipment or data, you assume all costs thereof.)',
    html.Br(),
    html.Br(),
    dcc.Markdown("""#### About"""),
    'Created by David J. Wu & Jean-Emmanuel Bibault - 2023',
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
    State('language', 'value'),
)
def update_table_results(n_clicks, *responses):
    if n_clicks == 0:
        return None, []

    questions = [
        'Number of Radiation treatments',
        'Increased passing of gas',
        'Diarrhea frequency',
        'Abdominal pain frequency',
        'Abdominal pain severity',
        'Abdominal pain with ADL',
        'Painful urination severity',
        'Urinary urgency frequency',
        'Urinary urgency with ADL',
        'Urinary frequency',
        'Urinary frequency with ADL',
        'Urine color change',
        'Urinary incontinence frequency',
        'Urinary incontinence with ADL',
        'Radiation skin reaction severity',
        'Fatigue severity',
        'Fatigue with ADL',
        'Additional symptoms',
    ]

    data = [{'question': question, 'answer': response} for question, response in zip(questions, responses)]
    language = responses[-1]
    summary = summarize_table(data,language)
    return summary, data

def summarize_table(data, language):
    messages = [{
        'role': 'system',
        'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
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
        model="gpt-4",
        messages=messages,
        n=1,
        stop=None,
        temperature=0.2,
    )

    summary = response.choices[0].message.content.strip()
    return summary

if __name__ == '__main__':
    app.run_server(debug=True)
