import numpy as np
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import tree
from yellowbrick.regressor import ResidualsPlot
from jinja2 import Environment, FileSystemLoader
from scipy import stats
import math
from jupyter_dash import JupyterDash
from sklearn.metrics import roc_curve
from dash import Dash, html, dcc, dash_table, callback
import plotly.express as px
import base64
import language_tool_python
import openai
import requests
import json

# Loading the folder that contains the txt templates

file_loader = FileSystemLoader('templates')

# Creating a Jinja Environment

env = Environment(loader=file_loader)

# Loading the Jinja templates from the folder
# For the regression
get_correlation = env.get_template('getcorrelation.txt')
model_comparison = env.get_template('modelcomparison.txt')
correlation_state = env.get_template('correlation.txt')
prediction_results = env.get_template('prediction.txt')
linearSummary = env.get_template('linearSummary.txt')
linearSummary2 = env.get_template('linearSummary2.txt')
linearSummary3 = env.get_template('linearSummary3.txt')
linearQuestion = env.get_template('linearQuestionset.txt')
DecisionTree1 = env.get_template('decisiontree1.txt')
DecisionTree2 = env.get_template('decisiontree2.txt')
DecisionTree3 = env.get_template('decisiontree3.txt')
DecisionTreeQuestion = env.get_template('decisiontreequestion.txt')
gamStory = env.get_template('gamStory.txt')
GAMslinear_stats = env.get_template('GAMsLinearL1')
GAMslinear_R2 = env.get_template('GAMsLinearL2')
GAMslinear_P = env.get_template('GAMsLinearL3')
GAMslinear_sum = env.get_template('GAMsLinearL4')
GB1 = env.get_template('GB1')
GB2 = env.get_template('GB2')
GB3 = env.get_template('GB3')

# For the classifier
logisticSummary = env.get_template('logisticSummary.txt')
logisticSummary2 = env.get_template('logisticSummary2')
logisticSummary3 = env.get_template('logisticSummary3.txt')
logisticQuestion = env.get_template('logisticQuestionset.txt')
classifieraccuracy=env.get_template('classifierAccuracy.txt')
classifierauc=env.get_template('classifierAUC.txt')
classifiercv=env.get_template('classifierCvscore.txt')
classifierf1=env.get_template('classifierF1score.txt')
classifierimp=env.get_template('classifierImportant.txt')
ridgequestionset=env.get_template('ridgeQuestionset.txt')
ridgedecision=env.get_template('ridgeDecision.txt')
classifierquestionset=env.get_template('classifierQuestionset.txt')
# For some basic function
basicdescription = env.get_template('basicdescription.txt')
simpletrend=env.get_template('simpletrend.txt')
modeloverfit=env.get_template('modeloverfit.txt')

# variables which each load a different segmented regression template
segmented_R2P = env.get_template('testPiecewisePwlfR2P')
segmented_R2 = env.get_template('testPiecewisePwlfR2')
segmented_P = env.get_template('testPiecewisePwlfP')
segmented_B = env.get_template('testPiecewisePwlfB')
segmented_GD1 = env.get_template('drugreport1')
segmented_GC1 = env.get_template('childreport1')

# For Aberdeen City CP
register_story = env.get_template('register.txt')
risk_factor_story = env.get_template('risk_factor.txt')
reregister_story = env.get_template('reregister.txt')
remain_story = env.get_template('remain_story.txt')
enquiries_story = env.get_template('enquiries_story.txt')

# For different dependent variables compared DRD
dc1 = env.get_template('dependentmagnificationcompare')
dc2 = env.get_template('samedependentmagnificationcompare')
dc3 = env.get_template('dependentquantitycompare')
dc4 = env.get_template('trendpercentagedescription')
dct = env.get_template('trenddescription')
tppc = env.get_template('twopointpeak_child')

# for different independent variables compared
idc1 = env.get_template('independentquantitycompare')
idtpc = env.get_template('independenttwopointcomparison')

# for batch processing
bp1 = env.get_template('batchprocessing1')
bp2 = env.get_template('batchprocessing2')

# for ChatGPT
databackground = env.get_template('databackground')
questionrequest=env.get_template('question_request.txt')

# for pycaret
automodelcompare1 = env.get_template('AMC1.txt')
automodelcompare2 = env.get_template('AMC2.txt')
pycaretimp = env.get_template('pycaret_imp.txt')
pycaretmodelfit = env.get_template('pycaret_modelfit.txt')
pycaretclassificationimp = env.get_template('pycaret_classificationimp.txt')
pycaretclassificationmodelfit = env.get_template('pycaret_classificationmodelfit.txt')
# for SKpipeline
pipeline_interpretation = env.get_template('pipeline_interpretation.txt')

# creating the global variables
models_names = ['Gradient Boosting Regressor', 'Random Forest Regressor', 'Linear Regression',
                'Decision Tree Regressor', 'GAMs']
models_results = []
g_Xcol = []
g_ycol = []
X_train = []
X_test = []
y_train = []
y_test = []
metricsData = DataFrame()
tmp_metricsData = DataFrame()


def MicroLexicalization(text):
    tool = language_tool_python.LanguageTool('en-US')
    # get the matches
    matches = tool.check(text)
    my_mistakes = []
    my_corrections = []
    start_positions = []
    end_positions = []

    for rules in matches:
        if len(rules.replacements) > 0:
            start_positions.append(rules.offset)
            end_positions.append(rules.errorLength + rules.offset)
            my_mistakes.append(text[rules.offset:rules.errorLength + rules.offset])
            my_corrections.append(rules.replacements[0])

    my_new_text = list(text)

    for m in range(len(start_positions)):
        for i in range(len(text)):
            my_new_text[start_positions[m]] = my_corrections[m]
            if (i > start_positions[m] and i < end_positions[m]):
                my_new_text[i] = ""
    my_new_text = "".join(my_new_text)
    return (my_new_text)


def GetCorrelation(data, Xcol, ycol):
    """This function takes in as input a dataset,the independent variables and the dependent variable, returning
    a story about the correlation between each independent variable and the dependent variable.

    :param data: This is the dataset that will be used in the analysis.
    :param Xcol: A list of independent variables.
    :param Ycol: The dependent/target variable.
    :return: A story about the correlation between Xcol and Ycol.
    """
    p_values = []
    coeff_values = []
    correlation = []
    independent_variables_number = 0
    for i in list(Xcol):
        coeff, p_value = stats.pearsonr(data[i], data[ycol])
        p_values.append(p_value)
        coeff_values.append(coeff)
        independent_variables_number += 1
        correlation.append(((data[[i, ycol]].corr())))

    for i in range(independent_variables_number):
        print(get_correlation.render(ycol=ycol, Xcol=Xcol[i], p_value=p_values[i], coeff_value=coeff_values[i]))
        plt.figure()
        sns.heatmap(correlation[i], annot=True, fmt='.2g', cmap='flare')  # graph only one correlation
        plt.show()


def FeatureSelection(data, ycol, threshold):
    """This function takes in as input a dataset,the dependent variable and the correlation treshold, returning ?

    :param data: This is the dataset that will be used in the analysis.
    :param Ycol: The dependent/target variable.
    :param treshold: This is the treshold that decides a significant correlation.
    :return:?
    """
    num_columns = data.select_dtypes(exclude='object').columns
    keep2 = []
    keep = []
    negative = []
    positive = []
    for i in list(num_columns):
        coeff, p_value = stats.pearsonr(data[i], data[ycol])
        if p_value < 0.05 and i != ycol:
            keep.append(i)
            if -1 < coeff < threshold * (-1):
                negative.append(i)
                keep2.append(i)
            elif threshold < coeff < 1:
                positive.append(i)
                keep2.append(i)
        # else :
        # del data[i]
    print(correlation_state.render(treshold=threshold, keep2=keep2, positive=positive, negative=negative))


def ModelData_view(mae_metrics, rmse_metrics, ycol):
    # Create DataFrame for the Model Metrics
    columns = {'MeanAbsoluteError': mae_metrics, 'RMSE': rmse_metrics}
    metricsData = DataFrame(data=columns, index=models_names)
    # Plot metrics data
    metricsData.plot(kind='barh', title='Model Comparison for Predictive Analysis', colormap='Pastel2')

    # Rank models and print comparison results
    metricsData['RankRMSE'] = metricsData['RMSE'].rank(method='min')
    metricsData['RankMAE'] = metricsData['MeanAbsoluteError'].rank(method='min')
    metricsData["rank_overall"] = metricsData[["RankRMSE", "RankMAE"]].apply(tuple, axis=1).rank(ascending=True)
    metricsData.sort_values(["rank_overall"], inplace=True, ascending=False)
    print(model_comparison.render(data=metricsData, yCol=ycol))


def Predict(model_type, values):
    global models_names, models_results, g_Xcol

    if len(values) != len(g_Xcol):
        print("The number of prediction values does not corespond with the number of predictive columns:")
        print("Required number of values is " + str(len(g_Xcol)) + "you put " + str(len(values)) + "values")
    else:
        prediction_value = models_results[models_names.index(model_type)].predict([values])
        print("Predicted Value is:" + str(prediction_value))
        print(prediction_results.render(xcol=g_Xcol, ycol=g_ycol, xcol_values=values, ycol_value=prediction_value,
                                        model_name=model_type, n=len(values)))


def display_story_dashboard():  # display comparison between models
    global models_names, models_results, X_train, y_train, X_test, y_test, metricsData
    metricsData_plot = metricsData.drop(columns=['RankRMSE', 'RankMAE', 'rank_overall'])
    fig = px.bar(metricsData_plot)
    story = model_comparison.render(data=metricsData, yCol=g_ycol)
    story_app = JupyterDash(__name__)

    story_app.layout = html.Div([dcc.Tabs([
        dcc.Tab(label='Comparison Chart', children=[
            dcc.Graph(figure=fig)
        ]),
        dcc.Tab(label='Data Story', children=[
            html.P(story)
        ]),
    ])
    ])
    story_app.run_server(mode='inline', debug=True)


def display_residual_dashboard():  # Risidual plots for models
    global models_names, models_results, X_train, y_train, X_test, y_test, metricsData
    _base64 = []
    for ind in metricsData.index:
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))

    residual_app = JupyterDash(__name__)

    residual_app.layout = html.Div([
        dcc.Tabs([
            dcc.Tab(label=metricsData.index[0], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[0]))
            ]),
            dcc.Tab(label=metricsData.index[1], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[1]))
            ]),
            dcc.Tab(label=metricsData.index[2], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[2]))
            ]),
            dcc.Tab(label=metricsData.index[3], children=[
                html.Img(src='data:image/png;base64,{}'.format(_base64[3]))
            ]),
        ])
    ])
    residual_app.run_server(mode='inline', debug=True)


def PrintGraphs(model_type='all'):
    global models_names, models_results, X_train, y_train, X_test, y_test, metricsData
    graphs = []
    if model_type == 'all':
        for ind in metricsData.index:
            current_index = models_names.index(ind)
            ysmodel = ResidualsPlot(models_results[current_index])
            ysmodel.fit(X_train, y_train)
            ysmodel.score(X_test, y_test)
            ysmodel.show(outpath='pictures/{}.png'.format(ind), clear_figure=True)

    else:
        current_index = models_names.index(model_type)
        ysmodel = ResidualsPlot(models_results[current_index])
        ysmodel.fit(X_train, y_train)
        ysmodel.score(X_test, y_test)
        ysmodel.show(outpath='pictures/{}.png'.format(current_index), clear_figure=True)


def start_app():
    app_name = JupyterDash(__name__)
    listTabs = []
    return (app_name, listTabs)

def data_background(Xcol,ycol,modelname=""):
    background=databackground.render(xcol=Xcol, ycol=ycol, modelname=modelname)
    return (background)

def set_chatGPT(Xcol,ycol,modelname,key,url="https://api.openai.com/v1/chat/completions"):
    openai.api_key = key
    url = url
    chatmodel="gpt-3.5-turbo"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    background = data_background(Xcol, ycol, modelname=modelname)
    messages = [{"role": "system", "content": background}, ]
    return (url,chatmodel,headers,messages)

def set_payload(message,messages=[]):
    messages.append( {"role": "user", "content": message}, )
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    return (payload,messages)

def send_response_receive_output(URL,headers,payload,messages):
    response = requests.post(URL, headers=headers, json=payload, stream=False)
    output=json.loads(response.content)["choices"][0]['message']['content']
    messages.append({"role": "assistant", "content": output})
    return (output,messages)

def run_app(app_name, listTabs,portnum=8050):
    app_name.layout = html.Div([dcc.Tabs(listTabs)])
    app_name.run_server(mode='inline', debug=True,port=portnum)

def read_figure(_base64, name):
    _base64.append(base64.b64encode(open('./{}.png'.format(name), 'rb').read()).decode('ascii'))
    return (_base64)

def scatterplot(Xdata,ydata,xlabel,ylabel):
    plt.scatter(Xdata, ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def data_basic_description(rownum,columnnum,strXcol,strycol):
    story = basicdescription.render(rownum=rownum,columnnum=columnnum,strXcol=strXcol,strycol=strycol)
    print(story)

def simple_trend(Xcol,ycol,Xforminy,Xformaxy,ave,miny,maxy):
    # Xforminy = ", ".join(map(str, Xforminy))
    # Xformaxy = ", ".join(map(str, Xformaxy))
    lengthXforminy=np.size(Xforminy)
    lengthXformaxy = np.size(Xformaxy)
    story = simpletrend.render(Xcol=Xcol,ycol=ycol,Xforminy=Xforminy,Xformaxy=Xformaxy,ave=ave,miny=miny,maxy=maxy,lenXmax=lengthXformaxy,lenXmin=lengthXforminy)
    print(story)


def LinearModelStats_view(data, Xcol, ycol, linearData, r2, questionset, expect,chatGPT=0,key="",portnum=8050):
    if expect=="":
        expect=["","",""]
    # Set for ChatGPT
    if chatGPT==1:
        URL, chatmodel, headers,messages=set_chatGPT(Xcol,ycol,modelname="linear",key=key)
        subsection="The following is the answer provided by ChatGPT:"

    # Store results for xcol
    for ind in linearData.index:
        ax = sns.regplot(x=ind, y=ycol, data=data)
        plt.savefig('pictures/{}.png'.format(ind))
        plt.clf()
    # Create Format index with file names
    _base64 = []
    for ind in linearData.index:
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))

    linear_app, listTabs = start_app()

    # Add to dashbord Linear Model Statistics
    fig = px.bar(linearData)
    question = linearQuestion.render(xcol=Xcol, ycol=ycol, qs=questionset, section=1, indeNum=np.size(Xcol),
                                     trend=expect[0])
    intro = linearSummary2.render(r2=r2, indeNum=np.size(Xcol), modelName="Linear Model", Xcol=Xcol,
                                  ycol=ycol, qs=questionset, t=expect[0],expect=expect[1])
    # intro = MicroLexicalization(intro)
    #set chatGPT
    aim = Xcol
    aim.insert(0, ycol)
    if chatGPT == 1:
        request=questionrequest.render(model="linear",question=1,r2information=str(round(r2,3)))
        payload, messages = set_payload(request + question, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.P(question), html.Br(), html.P(intro), html.Br(), html.P(subsection),html.Br(),html.P(output),
                dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    else:
        children = [html.P(question), html.Br(), html.P(intro),
                dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    dash_tab_add(listTabs, 'LinearModelStats', children)
    aim.remove(ycol)

    pf, nf, nss, ss, imp, i = "", "", "", "", "", 0
    # Add to dashbord Xcol plots and data story

    for ind in linearData.index:
        question = linearQuestion.render(xcol=ind, ycol=ycol, qs=questionset, section=2, indeNum=1, trend=expect[0])
        conflict = linearSummary.render(xcol=ind, ycol=ycol, coeff=linearData['coeff'][ind],
                                        p=linearData['pvalue'][ind], qs=questionset, expect=expect[2])

        # newstory = MicroLexicalization(story)
        if abs(linearData['coeff'][ind]) == max(abs(linearData['coeff'])):
            imp = ind
        if linearData['coeff'][ind] > 0:
            pf = pf + "the " + ind + ", "
        elif linearData['coeff'][ind] < 0:
            nf = nf + "the " + ind + ", "
        if linearData['pvalue'][ind] > 0.05:
            nss = nss + "the " + ind + ", "
        else:
            ss = ss + "the " + ind + ", "

        if questionset[1] == 1 or questionset[2] == 1:
            # set chatGPT
            if chatGPT == 1:
                request = questionrequest.render(model="linear", question=2, slopeinformation=str(round(linearData['coeff'][ind], 3)), pinformation=str(round(linearData['pvalue'][ind], 3)),ind=ind)
                payload, messages = set_payload(request + question,messages)
                output, messages = send_response_receive_output(URL, headers, payload, messages)
                print(output)
                children = [
                    html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                    html.P(conflict),html.Br(),html.P(subsection),html.Br(),html.P(output)
                ]
            else:
                children = [
                    html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                    html.P(conflict)
                ]
            dash_tab_add(listTabs, ind, children)

        i = i + 1
    question = linearQuestion.render(xcol="", ycol=ycol, qs=questionset, section=3, indeNum=1, trend=expect[0])
    summary = linearSummary3.render(imp=imp, ycol=ycol, nss=nss, ss=ss, pf=pf, nf=nf, t=expect[0], r2=r2,
                                    qs=questionset)
    if chatGPT == 1:
        payload, messages= set_payload(question,messages)
        output, messages = send_response_receive_output(URL, headers, payload, messages)
        print(output)
        children = [dcc.Graph(figure=fig), html.P(question), html.Br(), html.P(summary),html.Br(),html.P(subsection),html.Br(),html.P(output)]
    else:
        children = [dcc.Graph(figure=fig), html.P(question), html.Br(), html.P(summary)]
    dash_tab_add(listTabs, 'Summary', children)

    run_app(linear_app, listTabs,portnum)


def LogisticModelStats_view(data, Xcol, ycol, logisticData1, logisticData2, r2, questionset,chatGPT=0,key="",portnum=8050):
    # Store results for xcol
    for ind in logisticData1.index:
        ax = sns.regplot(x=ind, y=ycol, data=data, logistic=True)
        plt.savefig('pictures/{}.png'.format(ind))
        plt.clf()
    if chatGPT==1:
        URL, chatmodel, headers,messages=set_chatGPT(Xcol,ycol,modelname="Logistic",key=key)
        subsection="The following is the answer provided by ChatGPT:"
    # Create Format index with file names
    _base64 = []
    for ind in logisticData1.index:
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(ind), 'rb').read()).decode('ascii'))
    logistic_app, listTabs = start_app()
    i = 0

    # Add to dashbord Model Statistics
    question = logisticQuestion.render(indeNum=np.size(Xcol), xcol=Xcol, ycol=ycol, qs=questionset, section=1)
    intro = logisticSummary3.render(r2=r2, indeNum=np.size(Xcol), modelName="Logistic Model", Xcol=Xcol,
                                    ycol=ycol, qs=questionset, t=9)
    aim = Xcol
    aim.insert(0, ycol)
    if chatGPT == 1:
        request=questionrequest.render(model="logistic",question=1,ddofinformation=str(round(r2,3)))
        payload, messages = set_payload(request + question, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.P(question), html.Br(), html.P(intro), html.Br(), html.P(subsection),html.Br(),html.P(output),dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    # micro planning
    # intro = model.MicroLexicalization(intro)
    else:
        children = [html.P(question), html.Br(), html.P(intro),dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in
                                      data[aim].columns], style_table={'height': '400px', 'overflowY': 'auto'})]

    dash_tab_add(listTabs, 'LogisticModelStats', children)
    aim.remove(ycol)

    pos_eff, neg_eff, nss, ss, imp = "", "", "", "", ""
    # Add to dashbord Xcol plots and data story

    for ind in logisticData1.index:
        question = logisticQuestion.render(indeNum=1, xcol=ind, ycol=ycol, qs=questionset, section=2)
        # independent_variable_story
        independent_variable_story = logisticSummary.render(xcol=ind, ycol=ycol,
                                                            odd=abs(100 * (math.exp(logisticData1['coeff'][ind]) - 1)),
                                                            coeff=logisticData1['coeff'][ind],
                                                            p=logisticData1['pvalue'][ind],
                                                            qs=questionset)
        # independent_variable_story = model.MicroLexicalization(independent_variable_story)
        if logisticData1['coeff'][ind] == max(logisticData1['coeff']):
            imp = ind
        if logisticData1['coeff'][ind] > 0:
            pos_eff = pos_eff + ind + ", "
        else:
            neg_eff = neg_eff + ind + ", "
        if logisticData1['pvalue'][ind] > 0.05:
            nss = nss + ind + ", "
        else:
            ss = ss + ind + ", "
        if questionset[1] == 1 or questionset[2] == 1:
            if chatGPT == 1:
                request = questionrequest.render(model="logistic", question=2, coefinformation=str(
                    round(logisticData1['coeff'][ind], 3)), pinformation=str(round(logisticData1['pvalue'][ind], 3)),ind=ind)
                payload, messages = set_payload(request + question,messages)
                output, messages = send_response_receive_output(URL, headers, payload, messages)
                print(output)
                children = [
                    html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                    html.P(independent_variable_story),html.Br(),html.P(subsection),html.Br(),html.P(output)
                ]
            else:
                children = [html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                        html.P(independent_variable_story)]
            dash_tab_add(listTabs, ind, children)
        i = i + 1
    fig = px.bar(logisticData2)
    plt.savefig('pictures/{}.png'.format(imp))
    plt.clf()
    question = logisticQuestion.render(indeNum=1, xcol=ind, ycol=ycol, qs=questionset, section=3)
    summary = logisticSummary2.render(pos=pos_eff, neg=neg_eff, ycol=ycol, nss=nss, ss=ss, imp=imp,
                                      r2=r2, qs=questionset)
    # summary = model.MicroLexicalization(summary)
    if chatGPT == 1:
        payload, messages= set_payload(question,messages)
        output, messages = send_response_receive_output(URL, headers, payload, messages)
        print(output)
        children = [dcc.Graph(figure=fig), html.P(question), html.Br(), html.P(summary),html.Br(),html.P(subsection),html.Br(),html.P(output)]
    else:
        children = [dcc.Graph(figure=fig), html.P(question), html.Br(), html.P(summary)]
    dash_tab_add(listTabs, 'Summary', children)

    run_app(logistic_app, listTabs,portnum)

def RidgeClassifier_view(data,Xcol,ycol,rclf,pca,y_test, y_prob,roc_auc,X_pca,accuracy,importances,class1,class2):
    _base64 = []
    ridge_app, listTabs = start_app()
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('pictures/{}.png'.format("ROC"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("ROC"), 'rb').read()).decode('ascii'))
    plt.clf()

    # Plot decision boundaries
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_test, palette='husl')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500),
                         np.linspace(ylim[0], ylim[1], 500))
    Z = rclf.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.title('Decision Boundaries (Malignant vs. Benign)')
    plt.title('Decision Boundaries')
    plt.savefig('pictures/{}.png'.format("DecisionBoundaries"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("DecisionBoundaries"), 'rb').read()).decode('ascii'))

    # Create a dataframe to store feature importances along with their corresponding feature names
    X=data[Xcol]
    importances_df = DataFrame({'Feature': X.columns, 'Importance': importances})
    # Sort the dataframe by importance in descending order
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    # Plot feature importances using a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(importances_df['Feature'], importances_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=90)
    plt.savefig('pictures/{}.png'.format("FeatureImportances"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("FeatureImportances"), 'rb').read()).decode('ascii'))
    for i, importance in enumerate(importances):
        if abs(importance) == max(abs(importances)):
            #print("Feature {}: {}".format(X.columns[i], importance))
            imp=X.columns[i]

    # Extract coefficients from the trained model
    coefs = rclf.coef_[0]
    intercept = rclf.intercept_[0]

    # Construct the linear equation for the decision boundaries
    equation = 'Decision Boundary Equation: '
    for i in range(len(Xcol)):
        equation += '({:.4f} * {}) + '.format(coefs[i], Xcol[i])
    equation += '{:.4f}'.format(intercept)

    intro=classifieraccuracy.render(accuracy=round(accuracy,3),classifiername="ridge classifier")
    question=ridgequestionset.render(section=1)
    aim = Xcol
    aim.insert(0, ycol)
    children = [html.P(question), html.Br(), html.P(intro),dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in
                                      data[aim].columns], style_table={'height': '400px', 'overflowY': 'auto'})]

    dash_tab_add(listTabs, 'RidgeClassifierStats', children)
    aim.remove(ycol)

    question = ridgequestionset.render(section=2)
    DecisionBoundaryStory=ridgedecision.render(equation=equation,class1=class1,class2=class2)
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(question), html.Br(),
                html.P(DecisionBoundaryStory)]
    dash_tab_add(listTabs, "Decision Boundary Equation", children)

    question = ridgequestionset.render(section=3)
    aucStory=classifierauc.render(AUC=roc_auc)
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question), html.Br(),
                html.P(aucStory)]
    dash_tab_add(listTabs, "Area under the Receiver Operating Characteristic curve", children)

    question = ridgequestionset.render(section=4)
    ImpStory=classifierimp.render(imp=imp)
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question), html.Br(),
                html.P(ImpStory)]
    dash_tab_add(listTabs, "Feature Importances", children)

    run_app(ridge_app, listTabs)

def KNeighborsClassifier_view(data,Xcol,ycol,accuracy,precision,feature_importances,recall,f1,confusionmatrix,cv_scores):
    _base64 = []
    KNei_app, listTabs = start_app()
    # Print feature importances with column names
    for i in range(len(feature_importances)):
        if abs(feature_importances[i]) == max(abs(feature_importances)):
            print("Feature {}: {} - {:.2f}".format(i + 1, Xcol[i], feature_importances[i]))
            imp=Xcol[i]

    # Create a dictionary to store the evaluation metrics
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
    # Plot the evaluation metrics
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Evaluation Metrics")
    plt.savefig('pictures/{}.png'.format("Metrics"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("Metrics"), 'rb').read()).decode('ascii'))
    plt.clf()

    # Plot confusion matrix
    sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('pictures/{}.png'.format("confusionmatrix"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("confusionmatrix"), 'rb').read()).decode('ascii'))
    plt.clf()

    # Plot feature importances
    plt.barh(Xcol, feature_importances)
    plt.xlabel("Mutual Information")
    plt.title("Feature Importances")
    plt.savefig('pictures/{}.png'.format("imp"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("imp"), 'rb').read()).decode('ascii'))
    plt.clf()

    question=classifierquestionset.render(section=1)
    intro=classifieraccuracy.render(accuracy=round(accuracy,3),classifiername="K neighbors classifier")
    aim = Xcol
    aim.insert(0, ycol)
    children = [html.P(question), html.Br(), html.P(intro),dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in
                                      data[aim].columns], style_table={'height': '400px', 'overflowY': 'auto'})]

    dash_tab_add(listTabs, 'KNeighborsClassifierStats', children)
    aim.remove(ycol)

    question = classifierquestionset.render(section=2)
    modelStory=classifierf1.render(f1=round(f1,3))
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question), html.Br(),
                html.P(modelStory)]
    dash_tab_add(listTabs, "Model Evaluation Metrics", children)

    question = classifierquestionset.render(section=3)
    crossvalidStory=classifiercv.render(cv=round(cv_scores.mean(),3),cm=confusionmatrix)
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(question), html.Br(),
                html.P(crossvalidStory)]
    dash_tab_add(listTabs, "Confusion Matrix and Cross-validation", children)

    question = classifierquestionset.render(section=4)
    ImpStory=classifierimp.render(imp=imp)
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question), html.Br(),
                html.P(ImpStory)]
    dash_tab_add(listTabs, "Feature Importances", children)

    run_app(KNei_app, listTabs)

def SVCClassifier_view(data,Xcol,ycol,accuracy,precision,recall,f1,confusionmatrix,cv_scores):
    _base64 = []
    svm_app, listTabs = start_app()

    # Create a dictionary to store the evaluation metrics
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
    # Plot the evaluation metrics
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Evaluation Metrics")
    plt.savefig('pictures/{}.png'.format("Metrics"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("Metrics"), 'rb').read()).decode('ascii'))
    plt.clf()

    # Plot confusion matrix
    sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('pictures/{}.png'.format("confusionmatrix"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("confusionmatrix"), 'rb').read()).decode('ascii'))
    plt.clf()

    question=classifierquestionset.render(section=1)
    intro=classifieraccuracy.render(accuracy=round(accuracy,3),classifiername="Support Vector Machine model")
    aim = Xcol
    aim.insert(0, ycol)
    children = [html.P(question), html.Br(), html.P(intro),dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in
                                      data[aim].columns], style_table={'height': '400px', 'overflowY': 'auto'})]

    dash_tab_add(listTabs, 'SupportVectorMachineModelStats', children)
    aim.remove(ycol)

    question = classifierquestionset.render(section=2)
    modelStory=classifierf1.render(f1=round(f1,3))
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question), html.Br(),
                html.P(modelStory)]
    dash_tab_add(listTabs, "Model Evaluation Metrics", children)

    question = classifierquestionset.render(section=3)
    crossvalidStory=classifiercv.render(cv=round(cv_scores.mean(),3),cm=confusionmatrix)
    children = [html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(question), html.Br(),
                html.P(crossvalidStory)]
    dash_tab_add(listTabs, "Confusion Matrix and Cross-validation", children)

    run_app(svm_app, listTabs)

def kmeancluster_view(wcss,minnum_clusters,maxnum_clusters,summary,best_n_clusters,silhouette_score_value,calinski_harabasz_score_value,davies_bouldin_score_value):
    _base64 = []
    kmeancluster_app, listTabs = start_app()

    plt.plot(range(minnum_clusters, maxnum_clusters), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('pictures/{}.png'.format("ElbowMethod"))
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("ElbowMethod"), 'rb').read()).decode('ascii'))
    plt.clf()

    print(summary)
    print(best_n_clusters)
    print('Silhouette Score: {:.3f}'.format(silhouette_score_value))
    print('Calinski Harabasz Score: {:.3f}'.format(calinski_harabasz_score_value))
    print('davies bouldin score: {:.3f}'.format(davies_bouldin_score_value))

def TreeExplain(model, Xcol):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    explain = ""
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    explain = explain + (
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n ".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            explain = explain + (
                "{space}node={node} is a leaf node.\n".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            explain = explain + (
                "{space}node={node} is a split node: "
                "go to node {left} if {feature} <= {threshold} "
                "else to node {right}.\n".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=Xcol[feature[i]],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )
    return (explain)


def GradientBoostingModelStats_view(data, Xcol, ycol, GBmodel, mse, rmse, r2, imp, questionset, gbr_params,train_errors,test_errors,DTData,chatGPT=0,key="",portnum=8050):
    if chatGPT==1:
        URL, chatmodel, headers,messages=set_chatGPT(Xcol,ycol,modelname="Gradient Boosting",key=key)
        subsection="The following is the answer provided by ChatGPT:"

    # Store importance figure
    plt.bar(Xcol, GBmodel.feature_importances_)

    plt.title("Importance Score")
    plt.savefig('pictures/{}.png'.format("GB1"))
    plt.clf()

    # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    # export_graphviz(GBmodel.estimators_[5][0], out_file='pictures/small_tree.dot', feature_names=Xcol, rounded=True,
    #                 precision=1, node_ids=True)
    # (graph,) = pydot.graph_from_dot_file('pictures/small_tree.dot')
    # graph.write_png('pictures/small_tree.png', prog=['dot'])
    # encoded_image = base64.b64encode(open("pictures/small_tree.png", 'rb').read()).decode('ascii')

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(20, 10))  # Set the figure size as desired
    tree.plot_tree(GBmodel.estimators_[0][0], ax=ax,feature_names=Xcol,rounded=True,precision=1, node_ids=True)
    plt.savefig('pictures/tree_figure.png')
    encoded_image= base64.b64encode(open("pictures/tree_figure.png", 'rb').read()).decode('ascii')

    _base64 = []
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB1"), 'rb').read()).decode('ascii'))
    # Training & Test Deviance Figure
    test_score = np.zeros((gbr_params['n_estimators'],), dtype=np.float64)
    fig = plt.figure(figsize=(8, 8))
    plt.title('Deviance')
    plt.plot(np.arange(gbr_params['n_estimators']) + 1, GBmodel.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(gbr_params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.savefig('pictures/{}.png'.format("GB2"))
    plt.clf()
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB2"), 'rb').read()).decode('ascii'))

    plt.plot(train_errors, label='Training MSE')
    plt.plot(test_errors, label='Testing MSE')
    plt.legend()
    plt.savefig('pictures/{}.png'.format("GB3"))
    plt.clf()
    _base64.append(base64.b64encode(open('pictures/{}.png'.format("GB3"), 'rb').read()).decode('ascii'))

    GB_app, listTabs = start_app()
    # Add to dashbord Model Statistics
    question1 = DecisionTreeQuestion.render(q=1, m="gb")
    intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Gradient Boosting", Xcol=Xcol,
                                 ycol=ycol, )
    # micro planning
    # intro = model.MicroLexicalization(intro)
    aim = Xcol
    aim.insert(0, ycol)

    if chatGPT == 1:
        request=questionrequest.render(model="gradient boosting",question=1,r2information=str(round(r2,3)))
        payload, messages = set_payload(request + question1, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.P(question1), html.Br(), html.P(intro), html.Br(), html.P(subsection),html.Br(),html.P(output),dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]

    else:
        children = [html.P(question1), html.Br(), html.P(intro),
                dash_table.DataTable(data[aim].to_dict('records'), [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    dash_tab_add(listTabs, 'Gradient Boosting Stats', children)
    aim.remove(ycol)

    explain = TreeExplain(GBmodel.estimators_[0][0], Xcol)
    # listTabs.append(dcc.Tab(label="Training & Test Deviance", children=[
    #     html.Img(src='data:image/png;base64,{}'.format(_base64[1])), html.P(explain)
    # ]))
    question2 = DecisionTreeQuestion.render(q=2)
    if chatGPT == 1:
        request=questionrequest.render(model="gradient boosting",question=2,treeinformation=explain)
        payload, messages = set_payload(request + question2, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)), html.P(question2), html.Br(),html.Pre(explain), html.Br(),html.P(subsection),html.Br(),html.P(output)]
    else:
        children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)), html.P(question2), html.Br(),html.Pre(explain)]
    dash_tab_add(listTabs, 'Tree Explanation', children)

    summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=round(mse,3))
    question3 = DecisionTreeQuestion.render(q=3)

    if chatGPT == 1:
        request=questionrequest.render(model="gradient boosting",question=3,impinformation=str(DTData))
        payload, messages = set_payload(request + question3, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question3), html.Br(),html.P(summary), html.Br(),html.P(subsection),html.Br(),html.P(output)]
    else:
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[0])), html.P(question3), html.Br(),html.P(summary), ]
    dash_tab_add(listTabs, 'Summary', children)

    overfit=modeloverfit.render()
    question4=DecisionTreeQuestion.render(q=4)

    if chatGPT == 1:
        request=questionrequest.render(model="gradient boosting",question=4,trainerrorinformation=str(train_errors),testerrorinformation=str(test_errors))
        payload, messages = set_payload(request+ question4, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question4), html.Br(),html.P(overfit), html.Br(),html.P(subsection),html.Br(),html.P(output)]
    else:
        children = [html.Img(src='data:image/png;base64,{}'.format(_base64[2])), html.P(question4), html.Br(),html.P(overfit), ]
    dash_tab_add(listTabs, 'Model Fitting', children)

    run_app(GB_app, listTabs,portnum)


def RandomForestModelStats_view(data, Xcol, ycol, tree_small, rf_small, DTData, r2, mse, questionset,portnum=8050):
    # Save the tree as a png image

    # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    # export_graphviz(tree_small, out_file='pictures/small_tree.dot', feature_names=Xcol, rounded=True, precision=1,
    #                 node_ids=True)
    # (graph,) = pydot.graph_from_dot_file('pictures/small_tree.dot')
    # graph.write_png('pictures/small_tree.png', prog=['dot'])
    # encoded_image = base64.b64encode(open("pictures/small_tree.png", 'rb').read()).decode('ascii')

    # Extract one of the decision trees from the Random Forest model
    tree_idx = 0  # Index of the tree to visualize
    estimator = rf_small.estimators_[tree_idx]
    # Create a tree figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    tree.plot_tree(estimator, ax=ax, feature_names=Xcol, rounded=True, precision=1, node_ids=True)

    # Save the tree figure as a PNG image
    plt.savefig('pictures/tree_figure.png')
    # Encode the image as base64
    encoded_image = base64.b64encode(open("pictures/tree_figure.png", 'rb').read()).decode('ascii')


    # Explain of the tree
    explain = TreeExplain(rf_small.estimators_[0], Xcol)
    # Importance score Figure
    imp = ""
    fig = px.bar(DTData)
    for ind in DTData.index:
        if DTData['important'][ind] == max(DTData['important']):
            imp = ind
    # print(DTData['important'])
    RF_app, listTabs = start_app()
    # Add to dashbord Model Statistics
    intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Random Forest", Xcol=Xcol,
                                 ycol=ycol, )
    question1 = DecisionTreeQuestion.render(q=1, m="rf")
    # intro = MicroLexicalization(intro)
    aim = Xcol
    aim.insert(0, ycol)
    children = [html.P(question1), html.Br(), html.P(intro), dash_table.DataTable(data[aim].to_dict('records'),
                                                                                  [{"name": i, "id": i} for i in
                                                                                   data[aim].columns],
                                                                                  style_table={'height': '400px',
                                                                                               'overflowY': 'auto'})]
    dash_tab_add(listTabs, 'RandomForestModelStats', children)

    aim.remove(ycol)
    question2 = DecisionTreeQuestion.render(q=2)
    tree_explain_story = explain
    children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                html.P(question2), html.Br(), html.Pre(tree_explain_story)]
    dash_tab_add(listTabs, 'Tree Explanation', children)

    summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
    question3 = DecisionTreeQuestion.render(q=3)
    children = [dcc.Graph(figure=fig), html.P(question3), html.Br(), html.P(summary)]
    dash_tab_add(listTabs, 'Summary', children)

    run_app(RF_app, listTabs,portnum)


def DecisionTreeModelStats_view(data, Xcol, ycol, DTData, DTmodel, r2, mse, questionset,portnum=8050):
    # Importance score Figure
    imp = ""
    fig = px.bar(DTData)
    for ind in DTData.index:
        if DTData['important'][ind] == max(DTData['important']):
            imp = ind

    DT_app, listTabs = start_app()

    # Add to dashbord Model Statistics
    question1 = DecisionTreeQuestion.render(q=1, m="dt")
    intro = DecisionTree2.render(r2=r2, qs=questionset, indeNum=np.size(Xcol), modelName="Decision Tree", Xcol=Xcol,
                                 ycol=ycol, )
    # intro = MicroLexicalization(intro)
    aim = Xcol
    aim.insert(0, ycol)
    children = [html.P(question1), html.Br(), html.P(intro),
                dash_table.DataTable(data[aim].to_dict('records'), [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    dash_tab_add(listTabs,'DecisionTreeModelStats',children)
    aim.remove(ycol)

    # Figure of the tree
    fig2, axes = plt.subplots()
    tree.plot_tree(DTmodel,
                   feature_names=Xcol,
                   class_names=ycol,
                   filled=True,
                   node_ids=True);
    fig2.savefig('pictures/{}.png'.format("DT"))
    encoded_image = base64.b64encode(open("pictures/DT.png", 'rb').read()).decode('ascii')

    # # Text version of the tree node
    # feature_names_for_text = [0] * np.size(Xcol)
    # for i in range(np.size(Xcol)):
    #     feature_names_for_text[i] = Xcol[i]
    # explain = tree.export_text(DTmodel, feature_names=feature_names_for_text)
    # print(text_representation)
    # # Explain of the tree
    explain = TreeExplain(DTmodel, Xcol)
    # Text need to fix here
    tree_explain_story = explain
    question2 = DecisionTreeQuestion.render(q=2)
    children = [html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
                html.P(question2), html.Br(), html.Pre(tree_explain_story)]
    dash_tab_add(listTabs,'Tree Explanation',children)

    summary = DecisionTree3.render(imp=imp, ycol=ycol, r2=round(r2, 3), qs=questionset, mse=mse)
    question3 = DecisionTreeQuestion.render(q=3)
    children = [dcc.Graph(figure=fig), html.P(question3), html.Br(), html.P(summary)]
    dash_tab_add(listTabs,'Summary',children)

    run_app(DT_app,listTabs,portnum)


def GAMs_view(gam, data, Xcol, ycol, r2, p, conflict, nss, ss, mincondition, condition, questionset=[1, 1, 1, 0],
              trend=1,chatGPT=0,key="",predict="",portnum=8050):
    if chatGPT==1:
        URL, chatmodel, headers,messages=set_chatGPT(Xcol,ycol,modelname="generalized additive",key=key)
        subsection="The following is the answer provided by ChatGPT:"
    # Analysis and Graphs Generate
    _base64 = []
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        plt.plot(XX[:, term.feature], pdep)
        plt.plot(XX[:, term.feature], confi, c='r', ls='--')
        plt.title(Xcol[i])
        # plt.show()
        plt.savefig('pictures/{}.png'.format(i))
        _base64.append(base64.b64encode(open('pictures/{}.png'.format(i), 'rb').read()).decode('ascii'))
        plt.clf()
    # print(GAMslinear_R2.render(R=round(r2.get('explained_deviance'), 3), Xcol=Xcol, ycol=ycol,
    #                            indeNum=np.size(Xcol)))
    # print(GAMslinear_P.render(pvalue=p, Nss=nss, Ss=ss, Xcol=Xcol, ycol=ycol,
    #                           indeNum=np.size(Xcol)))
    # print(GAMslinear_sum.render(ycol=ycol, condition=condition, mincondition=mincondition, demand=1))

    gamm_app, listTabs = start_app()

    question = linearQuestion.render(xcol=Xcol, ycol=ycol, qs=questionset, section=1, indeNum=np.size(Xcol),
                                     trend=trend)
    # Add to dashbord GAMs Model Statistics
    intro = GAMslinear_stats.render(Xcol=Xcol, ycol=ycol, trend=0, indeNum=np.size(Xcol), r2=r2['McFadden_adj'])
    # Add table
    aim = Xcol
    aim.insert(0, ycol)
    # newstory = MicroLexicalization(story)

    if chatGPT == 1:
        payload, messages = set_payload("The R-aquared is "+ str(round(r2['McFadden_adj'],3)) +" and " + question, messages)
        output,messages=send_response_receive_output(URL,headers,payload,messages)
        print(output)
        children = [html.P(question), html.Br(), html.P(intro), html.Br(), html.P(subsection),html.Br(),html.P(output),
                dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    else:
        children = [html.P(question), html.Br(), html.P(intro), html.Br(),
                dash_table.DataTable(data[aim].to_dict('records'),
                                     [{"name": i, "id": i} for i in data[aim].columns],
                                     style_table={'height': '400px', 'overflowY': 'auto'})]
    dash_tab_add(listTabs, 'GAMs Model Stats', children)
    # dash_with_table_with_question(gamm_app, listTabs, question, intro, data[aim], 'GAMs Model Stats')
    # Fromat list with files names
    aim.remove(ycol)
    # Add to dashbord values of Xcol and graphs
    for i in range(len(Xcol)):
        question = linearQuestion.render(xcol=Xcol[i], ycol=ycol, qs=questionset, section=2, indeNum=1, trend=trend)
        # other story for one independent variable add in here
        story = gamStory.render(pvalue=p[i], xcol=Xcol[i], ycol=ycol, ) + conflict[i]
        if questionset[1] == 1 or questionset[2] == 1:
            # set chatGPT
            if chatGPT == 1:
                payload, messages = set_payload("The P-value of " + Xcol[i] + " is " + str(
                    round(p[i], 3)) +", and "+predict[i]+" Please briefly describe how the dependent variable changes as the independent variable changes, when does it reach the maximum and minimum values, and what is the trend? And whether the independent variable has a significant impact on the dependent variable?",messages)
                output, messages = send_response_receive_output(URL, headers, payload, messages)
                print(output)
                children = [
                    html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                    html.P(story),html.Br(),html.P(subsection),html.Br(),html.P(output)
                ]
            else:
                children = [
                    html.Img(src='data:image/png;base64,{}'.format(_base64[i])), html.P(question), html.Br(),
                    html.P(story)
                ]
            dash_tab_add(listTabs, Xcol[i], children)

        #dash_with_figure_and_question(gamm_app, listTabs, question, story, Xcol[i], _base64[i])

    question = linearQuestion.render(xcol="", ycol=ycol, qs=questionset, section=3, indeNum=1, trend=trend)
    summary = GAMslinear_P.render(pvalue=p, Nss=nss, Ss=ss, Xcol=Xcol, ycol=ycol,
                                  indeNum=np.size(Xcol)) + GAMslinear_sum.render(ycol=ycol, condition=condition,
                                                                                 mincondition=mincondition, demand=1)
    if chatGPT == 1:
        payload, messages= set_payload("Based on your previous answers only, without considering other elements. "+question,messages)
        output, messages = send_response_receive_output(URL, headers, payload, messages)
        print(output)
        children = [ html.P(question), html.Br(), html.P(summary),html.Br(),html.P(subsection),html.Br(),html.P(output)]
    else:
        children = [ html.P(question), html.Br(), html.P(summary)]
    dash_tab_add(listTabs, 'Summary', children)
    run_app(gamm_app,listTabs,portnum)

def dash_tab_add(listTabs, label, child):
    listTabs.append(dcc.Tab(label=label, children=child))


def dash_with_figure(app_name, listTabs, text, label, format, path='data:image/png;base64,{}'):
    listTabs.append(dcc.Tab(label=label, children=[
        html.Img(src=path.format(format)), html.P(text)
    ]))


def dash_with_figure_and_question(app_name, listTabs, question, text, label, format, path='data:image/png;base64,{}'):
    listTabs.append(dcc.Tab(label=label, children=[
        html.Img(src=path.format(format)), html.P(question), html.Br(), html.P(text)
    ]))


def dash_with_two_figure(app_name, listTabs, text, label, format1, format2, path='data:image/png;base64,{}'):
    listTabs.append(dcc.Tab(label=label, children=[
        html.Img(src=path.format(format1)), html.P(text), html.Img(src=path.format(format2))
    ]))


def dash_with_table(app_name, listTabs, text, dataset, label):
    listTabs.append(dcc.Tab(label=label,
                            children=[html.P(text),
                                      dash_table.DataTable(dataset.to_dict('records'),
                                                           [{"name": i, "id": i} for i in
                                                            dataset.columns],
                                                           style_table={'height': '400px',
                                                                        'overflowY': 'auto'})]), )


def dash_with_table_with_question(app_name, listTabs, question, text, dataset, label):
    listTabs.append(dcc.Tab(label=label,
                            children=[html.P(question), html.Br(), html.P(text),
                                      dash_table.DataTable(dataset.to_dict('records'),
                                                           [{"name": i, "id": i} for i in
                                                            dataset.columns],
                                                           style_table={'height': '400px',
                                                                        'overflowY': 'auto'})]), )


def dash_only_text(app_name, listTabs, text, label):
    listTabs.append(dcc.Tab(label=label,
                            children=[html.P(text), ]))


def dash_only_text_and_question(app_name, listTabs, question, text, label):
    listTabs.append(dcc.Tab(label=label,
                            children=[html.P(question), html.Br(), html.P(text), ]))


def register_question1_view(register_dataset, per1000inCity_col, diff, table_col, label, app, listTabs):
    registerstory = "The data from local comparators features in the Child Protection Register (CPR) report prepared quarterly. "
    i = 0
    for ind in per1000inCity_col:
        reslut = register_story.render(Xcol=ind, minX=min(register_dataset[ind]), maxX=max(register_dataset[ind]),
                                       diff=diff[i])
        registerstory = registerstory + reslut
        i = i + 1
    dash_with_table(app, listTabs, registerstory, register_dataset[table_col], label)


def riskfactor_question1_view(dataset, max_factor, same_factor, label, cityname, app, listTabs):
    riskstory = risk_factor_story.render(indeNum=(np.size(max_factor)), max_factor=max_factor,
                                         same_factor=same_factor,
                                         cityname=cityname)
    dash_with_table(app, listTabs, riskstory, dataset, label)


def re_register_question4_view(register_dataset, national_average_reregistration, reregister_lastyear, period,
                               table_col, label, app, listTabs):
    reregisterstory = reregister_story.render(nar=national_average_reregistration, rrly=reregister_lastyear,
                                              time=period)
    dash_with_table(app, listTabs, reregisterstory, register_dataset[table_col], label)


def remain_time_question5_view(remain_data, zero_lastdata, label, app, listTabs):
    remainstory = remain_story.render(zl=zero_lastdata)  # It can do more if I know the rule of answering this question
    dash_with_table(app, listTabs, remainstory, remain_data, label)


def enquiries_question6_view(ACmean, ASmean, MTmean, ACdata, ASdata, MTdata, period, label, app, listTabs):
    enquiriesstory = enquiries_story.render(indeNum=(np.size(period)), ACM=ACmean, ASM=ASmean, MTM=MTmean,
                                            ACE=ACdata,
                                            ASE=ASdata,
                                            MTE=MTdata, period=period)
    dash_only_text(app, listTabs, enquiriesstory, label)


def segmentedregressionsummary_CPview(X, ymax, Xmax, ylast, Xlast, diff1, diff2, Xbegin, Xend, yend, iP, dP, nP, Xcol,
                                      ycol):
    print(segmented_GC1.render(
        X=X,
        ymax=ymax,
        Xmax=Xmax,
        ylast=ylast,
        Xlast=Xlast,
        diff1=diff1,
        diff2=diff2,
        Xbegin=Xbegin,
        Xend=Xend,
        yend=yend,
        iP=iP,
        dP=dP,
        nP=nP,
        Xcol=Xcol,
        ycol=ycol, ))


def segmentedregressionsummary_DRDview(increasePart, decreasePart, notchangePart, ycolname, maxIncrease, maxDecrease):
    print(segmented_GD1.render(
        iP=increasePart,
        dP=decreasePart,
        nP=notchangePart,
        ycol=ycolname,
        mI=maxIncrease,
        mD=maxDecrease, ))


def dependentcompare_view(Xcolname, begin, end, ycolname1, ycolname2, magnification1, magnification2, X, X1, X2):
    print(dc1.render(Xcol=Xcolname, begin=begin, end=end, loopnum=end, y1name=ycolname1, y2name=ycolname2,
                     magnification1=magnification1,
                     magnification2=magnification2, X=X, X1=X1, X2=X2))


def batchprovessing_view1(m, Xcolname, X1, X2, y, allincrease, alldecrease, category_name, ycolnames, begin, end):
    story = (bp1.render(mode=m, Xcol=Xcolname, X1=0, allincrease=allincrease, alldecrease=alldecrease,
                        category_name=category_name)) + "\n"
    for i in range(np.size(ycolnames) - 1):
        ycolname = ycolnames[i]
        ydata = y[ycolname]
        y1 = ydata[begin]
        y2 = ydata[end]
        story = story + bp2.render(mode=m, ycol=ycolname, y1=y1, y2=y2, X1=X1, X2=X2, mag=0)
    print(story)


def batchprovessing_view2(m, Xcolname, X1, allincrease, alldecrease, category_name, total, ycolnames, y, point):
    story = (bp1.render(mode=m, Xcol=Xcolname, X1=X1, allincrease=False, alldecrease=False,
                        category_name=category_name)) + "\n"
    for i in range(np.size(ycolnames) - 1):
        ycolname = ycolnames[i]
        ydata = y[ycolname]
        y1 = ydata[point]
        mag = np.round(y1 / total, 2)
        story = story + bp2.render(mode=m, ycol=ycolname, y1=y1, y2=0, X1=0, X2=0, mag=mag)
    print(story)


def independenttwopointcompare_view(Xcolname, point, ycolname1, ycolname2, X, y1, y2, mode, mag):
    print(idtpc.render(Xcol=Xcolname, point=point, y1name=ycolname1, y2name=ycolname2, X=X, y1=y1, y2=y2,
                       mode=mode, mag=mag))


def two_point_and_peak_child_view(Xcolname, ycolname, Xpeak, ypeak, X1, X2, y1, y2):
    print(tppc.render(Xcol=Xcolname, ycol=ycolname, Xpeak=Xpeak, ypeak=ypeak, X1=X1, X2=X2, y1=y1, y2=y2))


def trendpercentage_view(Xcolname, begin, end, ycolname, X, y, std, samepoint):
    print(dc4.render(Xcol=Xcolname, begin=begin, end=end, ycol=ycolname, X=X, y=y, std=std, samepoint=samepoint))


def pycaret_find_one_best_model(model, detail, n, sort, exclude, excludeNum):
    print(automodelcompare1.render(best=model, detail=detail, n_select=n, sort=sort, exclude=exclude,
                                   excludeNum=excludeNum))
    modelcomparestory = automodelcompare1.render(best=model, detail=detail, n_select=n, sort=sort, exclude=exclude,
                                                 excludeNum=excludeNum)
    return (modelcomparestory)


def pycaret_find_best_models(model, detail, n, sort, exclude, excludeNum, length):
    print(automodelcompare2.render(best=model, detail=detail, n_select=n, sort=sort, exclude=exclude, length=length,
                                   excludeNum=excludeNum))
    modelcomparestory = automodelcompare2.render(best=model, detail=detail, n_select=n, sort=sort, exclude=exclude,
                                                 length=length, excludeNum=excludeNum)
    return (modelcomparestory)


def pycaret_model_summary_view(imp_var, r2, mape, imp_pos_ave, imp_pos_value_ave, imp_neg_ave, imp_neg_value_ave,
                               target):
    story1 = pycaretmodelfit.render(r2=r2, mape=mape)
    story2 = pycaretimp.render(imp=imp_var, target=target, imp_pos_ave=imp_pos_ave, imp_pos_value_ave=imp_pos_value_ave,
                               imp_neg_ave=imp_neg_ave, imp_neg_value_ave=imp_neg_value_ave)
    print(story1)
    print(story2)
    return (story1, story2)


def pycaret_classification_model_summary_view(imp_var, r2, mape, imp_pos_ave, imp_pos_value_ave, imp_neg_ave,
                                              imp_neg_value_ave, target):
    story1 = pycaretclassificationmodelfit.render(r2=r2, mape=mape)
    story2 = pycaretclassificationimp.render(imp=imp_var, target=target, imp_pos_ave=imp_pos_ave,
                                             imp_pos_value_ave=imp_pos_value_ave, imp_neg_ave=imp_neg_ave,
                                             imp_neg_value_ave=imp_neg_value_ave)
    print(story1)
    print(story2)
    return (story1, story2)


def skpipeline_interpretation(pipelinename):
    story = pipeline_interpretation.render(pipe=pipelinename)
    # print(story)
    return (story)
