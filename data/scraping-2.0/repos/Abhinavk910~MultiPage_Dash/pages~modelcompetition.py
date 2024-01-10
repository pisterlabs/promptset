
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:55:11 2023

@author: abhinav.kumar
"""

import dash_mantine_components as dmc
from dash import Output, Input, State, ALL, no_update, dcc, ctx
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import dash
import numpy as np
import time
import pandas as pd
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import AI21
from langchain.llms import Replicate

from helping_functions.model_competition_fx import (
                        get_model_grid, create_acccordion_item,create_card_p8,
                        get_answer_openai, get_answer_ai21, get_answer_replicate
                    ) 
from helping_pages.model_competition_layout import layout_mc

dash.register_page(__name__)

layout = layout_mc


#CB 0: Handling interval and animation
dash.clientside_callback(
    """
    function updatetext(n) {
        if (n == 1) {
            return [true, false, "Model Competition"]
        } else {
        return [false, false, n];
      }
    }
    """,
    Output("html-div-to-hide", "hidden"),
    Output('html-div-to-hide2', 'hidden'),
    Output('header-text-p8', 'children'),
    Input("interval1-p8", "n_intervals"),
    prevent_initial_call=True,
)

            
#CB 1: Selecting number of Competitors
@dash.callback(
    Output('competitor-selection-p8', 'children'),
    Input('competitor-count-p8', 'value')
)
def update_competitor_input(c):
    # Callback Trigger on render or competitor value change.
    # Creating saperate dropdown so that one can select model from it
    # add the end create a invisible button(display none) "Next"
    
    model_grid = list()
    for i in range(1, c+1):
        id1 = {'type':'llm-name-p8', 'index':i}
        id2 = {'type':'llm-temp-p8', 'index':i}
        model_grid.append(dmc.Divider(label=f'Competitor-{i}', labelPosition='center'))
        model_grid.append(get_model_grid(i, id1, id2))
        
    model_grid.append(
        dmc.Center(
            dmc.Button(
                id='register-api-accordian-p8',
                leftIcon=DashIconify(icon='carbon:next-outline'),
                children=['Next'],
                variant='gradient',
                style={'width':'200px', 'display':'none'}
            )
        )
    )
    return model_grid


#CB2 - Handle invisible btn that added in CB1.
@dash.callback(
    Output('register-api-accordian-p8', 'style'),
    Input({"type": "llm-name-p8", "index": ALL}, "value")
)
def check_all_competitor_filled(value_):
    # it will check the value of all dropdown of competitor
    # if all selected than invisiblity of btn gone.(display : "")
    if None in value_:
        return no_update
    else:
        return {'width':'200px', 'display':''}


#CB3 - Handle Next btn Click
@dash.callback(
    Output('competition-accordion-p8', 'style'),
    Output('competition-accordion-api-p8', 'style', allow_duplicate=True),
    Output('competition-accordion-api-p8', 'children'),
    Output('competition-data-p8', 'data', allow_duplicate=True),
    Input('register-api-accordian-p8', 'n_clicks'),
    State('competition-data-p8', 'data'),
    State('competitor-count-p8', 'value'),
    State({"type": "llm-name-p8", "index": ALL}, "value"),
    State({"type": "llm-temp-p8", "index": ALL}, "value"),
    prevent_initial_call=True
)
def on_clicking_next_btn(n, data, count_, model_, temp_):
    # We will to hide the competitor selection accordion
    # Will unhide api key register accordioin and populate the children
    # will store some data.
    
    if n:
        #saving competitor count, models from dropdown and corresponding temperature.
        data['contender-count'] = count_
        data['$contender-list'] = model_
        data['$contender-temp'] = temp_
        
        #Extracting LLM Provide, saving llm and api key in data 
        llm = list(set([i.split(' - ')[0].strip() for i in model_]))
        data['llms'] = sorted(llm)
        data['llms-api_registered'] = [False]*len(llm)
        api_accordion_list = list()
        for i, j in enumerate(sorted(llm)):
            id1 = {'type':'llm-api-password-p8', 'index':i}
            id2 = {'type':'llm-api-btn-p8', 'index':i}
            id3 = {'type':'api-alert-success-p8', 'index':i}
            id4 = {'type':'api-alert-failure-p8', 'index':i}
            api_accordion_list.append(create_acccordion_item(value=f'{j}-key-p8',
                                                             control_name=f'{j} Key Register',
                                                             password_id=id1,
                                                             button_id=id2,
                                                             alert1_id=id3,
                                                             alert2_id=id4,
                                                             api_description=''))
        
        api_accordion_list.append(
            dmc.Center(
                dmc.Button(
                    id='next2-p8',
                    leftIcon=DashIconify(icon='carbon:next-outline'),
                    children=['Next'],
                    variant='gradient',
                    style={'width':'200px', 'display':'none'}
                )
            )
        )
        
        
        # this is dataframe created so that all question and answer will be store accordingly
        columns = list()

        columns.append('question')

        for i in range(count_):
            column = [f'contender-{i+1}',f'answer-{i+1}',f'timing-{i+1}']
            columns += column

        df = pd.DataFrame(columns=columns, data=[[1]*len(columns)])
        data['all_data'] = df.to_dict('records')
        
        return {'display':'none'}, {'display':''}, api_accordion_list, data
    else:
        raise PreventUpdate


#CB4 - Handle loading State of register Btn.
dash.clientside_callback(
    """
    function updateLoadingState(n_clicks) {
          var length = n_clicks.length;
          var trueArray = Array(length).fill(true);
          return trueArray;
    }
    """,
    Output({"type": "llm-api-btn-p8", "index": ALL}, "loading", allow_duplicate=True),
    Input({"type": "llm-api-btn-p8", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)


#CB5 - Handle Register Btns Click
@dash.callback(
    Output({"type": "api-alert-success-p8", "index": ALL}, "hide"),
    Output({"type":"api-alert-failure-p8", "index":ALL}, "hide"),
    Output({"type":"llm-api-btn-p8", "index":ALL}, "loading"),
    Output({"type":"api-alert-failure-p8", "index":ALL}, "children"),
    Output("competition-data-p8", "data", allow_duplicate=True),
    Output("next2-p8", 'style'),
    Output({"type": "llm-api-btn-p8", "index": ALL}, "gradient"),
    Output({"type": "llm-api-btn-p8", "index": ALL}, "leftIcon"),
    Input({"type": "llm-api-btn-p8", "index": ALL}, "n_clicks"),
    State({"type": "llm-api-password-p8", "index": ALL}, "value"),
    State("competition-data-p8", "data"),
    State({"type": "llm-api-btn-p8", "index": ALL}, "gradient"),
    State({"type": "llm-api-btn-p8", "index": ALL}, "leftIcon"),
    prevent_initial_call=True
)
def register_api(n_, api_key_, data, grad_, lefticon_):
    # First check the api key on llm provider.
    # if correct then will update alearts, register btn color, register btn icon
    # also if all the required provider api key resgistered then Next2 Btn will be visible
    
    # arg is the index value of the register btn clicked
    arg = ctx.triggered_id['index']
    n_ = [0 if i is None else i for i in n_]
    api_key = ['no_update' if i is None or i == "" else i for i in api_key_]
    api_key = api_key_[arg]
    model = data['llms'][arg].lower()
    output_ = [True]*len(n_)
    output2_ = [True]*len(n_)
    loading = [False]*len(n_)
    failure_text = ['no']*len(n_)
    default_style = {'width':'200px', 'display':'none'}
    
    prompt = PromptTemplate(
                    input_variables=["product"],
                    template="What is a good name for a company that makes {product}?",
                 )
    if sum(n_) == 0:
        raise PreventUpdate
    else:

        if model == 'openai':
            try:
                chatopenai = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key, temperature=0)
                llmchain_chat = LLMChain(llm=chatopenai, prompt=prompt)
                llmchain_chat.run("colorful socks")
                data['llms-api_registered'][arg] = True
                
                if False not in data['llms-api_registered']:
                    default_style = {'width':'200px', 'display':''}
                output_[arg]=False
                grad_[arg]={"from": "teal", "to": "lime", "deg": 105}
                lefticon_[arg]=DashIconify(icon="icon-park-outline:correct")
                data['keys'][model]=api_key
                return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_
            except Exception as e:
                output2_[arg]=False
                failure_text[arg] = str(e)
                return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_
        elif model == 'ai21':
            try:
                llm = AI21(ai21_api_key=api_key, temperature=1)
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                llm_chain.run("pant")
                data['llms-api_registered'][arg] = True
                if False not in data['llms-api_registered']:
                    default_style = {'width':'200px', 'display':''}
                grad_[arg]={"from": "teal", "to": "lime", "deg": 105}
                output_[arg]=False
                lefticon_[arg]=DashIconify(icon="icon-park-outline:correct")
                data['keys'][model]=api_key
                return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_
            except Exception as e:
                output2_[arg] = False
                failure_text[arg] = str(e)
                return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_
        elif model == 'replicate':
            try:
                os.environ["REPLICATE_API_TOKEN"] = api_key
                llm = Replicate(model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b")
                text = llm('what is 2+2?')
                data['llms-api_registered'][arg] = True
                if False not in data['llms-api_registered']:
                    default_style = {'width':'200px', 'display':''}
                grad_[arg]={"from": "teal", "to": "lime", "deg": 105}
                output_[arg]=False
                lefticon_[arg]=DashIconify(icon="icon-park-outline:correct")
                data['keys'][model]=api_key
                return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_
            except Exception as e:
                output2_[arg] = False
                failure_text[arg] = str(e)
                return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_
        else:
            output2_[arg] = False
            failure_text[arg] = str(model)
            return output_, output2_, loading, failure_text, data, default_style, grad_, lefticon_

        
#CB6 - Handle Next2 Btn click. Will create cards for competitors
@dash.callback(
    Output('compete-page-p8', 'hidden'),
    Output('competitor-card-p8', 'children'),
    Output('competition-accordion-api-p8', 'style',allow_duplicate=True),
    Output("competition-data-p8", "data", allow_duplicate=True),
    Input('next2-p8', 'n_clicks'),
    State("competition-data-p8", "data"),
    prevent_initial_call=True
)
def on_clicking_next2_btn(nc, data):
    #prevent initial call not working so nc is used here
    # after click on btn, card with competitor will occur
    if nc:
        model_ = data['$contender-list']
        temp_ = data['$contender-temp']

        competitor_list = list()
        for i, (model, temp) in enumerate(zip(model_, temp_)):
            id1 = {'type':'competitor-timing-p8', 'index':i}
            id2 = {'type':'competitor-text-p8', 'index':i}
            id3 = {'type':'competitor-card-p8', 'index':i}
            id4 = {'type':'competitor-score-p8', 'index':i}
            competitor_list.append(create_card_p8(i, model, temp, id1, id2, id3, id4))
        
        data['answered_question'] = 0
        data['current_question']  = 0
        return False, competitor_list, {'display':'none'}, data
    else:
        raise PreventUpdate


#CB7 - Handle loading State of ask btn
dash.clientside_callback(
    """
    function updateLoadingState(n_clicks) {
          return true;
    }
    """,
    Output("answer-output-p8", "loading", allow_duplicate=True),
    Input("answer-output-p8", "n_clicks"),
    prevent_initial_call=True,
)


#CB8 - Handle Ask btn click
@dash.callback(
    Output({"type": "competitor-timing-p8", "index": ALL}, "children"),
    Output({"type": "competitor-timing-p8", "index": ALL}, "color"),
    Output({"type": "competitor-text-p8", "index": ALL}, "value"),
    Output({"type": "competitor-score-p8", "index": ALL}, "children"),
    Output("answer-output-p8", "loading"),
    Output("answer-output-p8", "style", allow_duplicate=True),
    Output('next-btn-p8', 'style', allow_duplicate=True),
    Output("competition-data-p8", "data", allow_duplicate=True),
    Input('answer-output-p8', 'n_clicks'),
    State("competition-data-p8", "data"),
    State({"type": "competitor-score-p8", "index": ALL}, "children"),
    State('question-input-p8', 'value'),
    prevent_initial_call=True
)
def competitor_answer(n, data, score_, question):
    # will update all the competitors card with their answer
    # also update time required in to complete the answer
    # also keep track of the score.
    
    model_ = data['$contender-list']
    temp_ = data['$contender-temp']
    
    llm_parent = [i.split(" - ")[0].lower() for i in model_]
    llm_children = [i.split(" - ")[1] for i in model_]
    
    time_ = list()
    text_ = list()
    color_ = ['red']*len(llm_parent)

    if question == "":
        question = 'You are a contender in model competition, just say hi!!'
    
    for i, (llm, model, temp) in enumerate(zip(llm_parent, llm_children, temp_)):
        if llm == 'openai':
            start = time.time()
            api_key = data['keys'][llm]
            answer = get_answer_openai(question, model, api_key, temp)
            end = time.time()
            interval = end - start
            time_.append(interval)
            text_.append(str(answer))
        elif llm == 'ai21':
            start=time.time()
            api_key = data['keys'][llm]
            answer = get_answer_ai21(question, model, api_key, temp)
            end=time.time()
            interval = end - start
            time_.append(interval)
            text_.append(str(answer))
        elif llm == 'replicate':
            start=time.time()
            api_key = data['keys'][llm]
            os.environ["REPLICATE_API_TOKEN"] = api_key
            if model == 'vicuna-13b':
                model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
            elif model == 'stability-ai/stablelm-tuned-alpha-7b':
                model="stability-ai/stablelm-tuned-alpha-7b:c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb"
            elif model == 'llama-7b':
                model="replicate/llama-7b:ac808388e2e9d8ed35a5bf2eaa7d83f0ad53f9e3df31a42e4eb0a0c3249b3165"
            elif model == 'dolly-v2-12b':
                model="replicate/dolly-v2-12b:ef0e1aefc61f8e096ebe4db6b2bacc297daf2ef6899f0f7e001ec445893500e5"
            elif model == 'oasst-sft-1-pythia-12b':
                model="replicate/oasst-sft-1-pythia-12b:28d1875590308642710f46489b97632c0df55adb5078d43064e1dc0bf68117c3"
            elif model == 'gpt-j-6b':
                model="replicate/gpt-j-6b:b3546aeec6c9891f0dd9929c2d3bedbf013c12e02e7dd0346af09c37e008c827"
            
            answer = get_answer_replicate(question, model, api_key, temp)
            end=time.time()
            interval = end - start
            time_.append(interval)
            text_.append(str(answer))

            
    arg = np.argmin(time_) 
    color_[arg]='green'
    time_ = [f"Timing - {round(time, 2)} sec" for time in time_]
    
    if score_[arg] != " ":
        s = int(score_[arg].split(' - ')[-1]) + 1
        score_[arg] = f"Score - {s}"
    else:
        score_[arg] = "Score - 1"
        
    columns = list()
    columns.append(question)

    for i, j, k in zip(model_, text_, time_):
        list_ = [i, j , k]
        columns += list_

    df = pd.DataFrame(data['all_data'])
    df2 = pd.DataFrame(data=[columns], columns=list(df.columns))
    data['all_data'] = pd.concat([df, df2]).to_dict('records')
    data['answered_question'] += 1 
    
    return time_, color_, text_, score_, False, {'display':'none'}, {'display':''}, data


#CB9 - Handle loading State of next btn
dash.clientside_callback(
    """
    function updateLoadingState(n_clicks) {
          return true;
    }
    """,
    Output("next-btn-p8", "loading", allow_duplicate=True),
    Input("next-btn-p8", "n_clicks"),
    prevent_initial_call=True,
)


#CB10 - Handle Next btn click after we got answer
@dash.callback(
    Output({"type": "competitor-timing-p8", "index": ALL}, "children", allow_duplicate=True),
    Output({"type": "competitor-text-p8", "index": ALL}, "value", allow_duplicate=True),
    Output('question-input-p8', 'value'),
    Output("next-btn-p8", "loading"),
    Output("answer-output-p8", "style", allow_duplicate=True),
    Output('next-btn-p8', 'style', allow_duplicate=True),
    Output('back-question-p8', 'disabled', allow_duplicate=True),
    Output('next-question-arrow-p8', 'disabled', allow_duplicate=True),
    Output("competition-data-p8", "data", allow_duplicate=True),
    Input('next-btn-p8', 'n_clicks'),
    State({"type": "competitor-text-p8", "index": ALL}, "value"),
    State("competition-data-p8", "data"),
    prevent_initial_call=True
)
def next_btn_click(n, comp_, data):
    data['current_question'] += 1
    return [' ']*len(comp_), [' ']*len(comp_), '',False, {'display':''}, {'display':'none'}, False, True, data
    

#CB11 - Handle ActionIcon "Back" click   
@dash.callback(
    Output({"type": "competitor-timing-p8", "index": ALL}, "children", allow_duplicate=True),
    Output({"type": "competitor-text-p8", "index": ALL}, "value", allow_duplicate=True),
    Output('question-input-p8', 'value', allow_duplicate=True),
    Output("answer-output-p8", "style"),
    Output('next-btn-p8', 'style'),
    Output('back-question-p8', 'disabled'),
    Output('next-question-arrow-p8', 'disabled'),
    Output("competition-data-p8", "data", allow_duplicate=True),
    Input('back-question-p8', 'n_clicks'),
    State("competition-data-p8", "data"),
    prevent_initial_call=True
)
def history_back_btn_click(n, data):
    # will show history of question and reply by the competitor
    
    q_no = data['current_question'] - 1
    
    if q_no < 1:
        q_no = 1
        data['current_question'] = 0
    else:
        data['current_question'] = q_no
    back_disable=False
    
    if q_no == 1:
        back_disable = True
    
    dd = pd.DataFrame(data['all_data'])
    dd.reset_index(inplace=True, drop=True)
    da_ = dd.iloc[q_no]
    question = da_[0]
    contender_ = list()
    answer_=list()
    time_ = list()
    for i in range(1, len(da_)):
        print(i%3)
        if i%3 == 1:
            contender_.append(da_[i])
        elif i%3 == 2:
            answer_.append(da_[i])
        else:
            time_.append(da_[i])

    return time_, answer_, question, {'display':'none'}, {'display':'none'}, back_disable, False, data


# CB12 - Handle ActionIcon "Next" click
@dash.callback(
    Output({"type": "competitor-timing-p8", "index": ALL}, "children", allow_duplicate=True),
    Output({"type": "competitor-text-p8", "index": ALL}, "value", allow_duplicate=True),
    Output('question-input-p8', 'value', allow_duplicate=True),
    Output("answer-output-p8", "style", allow_duplicate=True),
    Output('next-btn-p8', 'style', allow_duplicate=True),
    Output('back-question-p8', 'disabled', allow_duplicate=True),
    Output('next-question-arrow-p8', 'disabled', allow_duplicate=True),
    Output("competition-data-p8", "data", allow_duplicate=True),
    Input('next-question-arrow-p8', 'n_clicks'),
    State("competition-data-p8", "data"),
    prevent_initial_call=True
)
def history_next_btn_click(n, data):
    # to navigate history of question.
    
    q_no = data['current_question'] + 1
    data['current_question'] = q_no
    next_disable=False
    next_btn_display = {'display':'none'}
    if q_no == data['answered_question']:
        next_disable = True
        next_btn_display = {'display':''}
    
    dd = pd.DataFrame(data['all_data'])
    dd.reset_index(inplace=True, drop=True)
    da_ = dd.iloc[q_no]
    question = da_[0]
    contender_ = list()
    answer_=list()
    time_ = list()
    for i in range(1, len(da_)):
        print(i%3)
        if i%3 == 1:
            contender_.append(da_[i])
        elif i%3 == 2:
            answer_.append(da_[i])
        else:
            time_.append(da_[i])

    return time_, answer_, question, {'display':'none'}, next_btn_display, False, next_disable, data


#CB13 - Reset action icon
dash.clientside_callback(
    """
    function updatetext(n) {
        return [{'display':''}, {'display':'none'}, true]
    }
    """,
    Output("competition-accordion-p8", "style", allow_duplicate=True),
    Output('competition-accordion-api-p8', 'style', allow_duplicate=True),
    Output('compete-page-p8', 'hidden', allow_duplicate=True),
    Input("back-to-start", "n_clicks"),
    prevent_initial_call=True,
)


#CB14 - Download
@dash.callback(
    Output("download-xslx-p8", "data"), 
    Output('no-question-asked-p8', 'hide'),
    Input("btn-xslx-p8", "n_clicks"),
    State('competition-data-p8', 'data'),
    prevent_initial_call=True
)
def generate_xlsx(n_nlicks, data):
    
    if data['answered_question'] != 0:
        df = pd.DataFrame(data['all_data'])
        df.reset_index(inplace=True, drop=True)
        df = df.iloc[1:]
        def to_xlsx(bytes_io):
            xslx_writer = pd.ExcelWriter(bytes_io, engine="xlsxwriter")  # requires the xlsxwriter package
            df.to_excel(xslx_writer, sheet_name="sheet1")
            xslx_writer.save()

        return dcc.send_bytes(to_xlsx, "data.xlsx"), True
    else:
        return no_update, False









