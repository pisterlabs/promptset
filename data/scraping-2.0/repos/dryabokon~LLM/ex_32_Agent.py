import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_interaction
from LLM2 import llm_config
from LLM2 import llm_models
from LLM2 import llm_tools
from LLM2 import llm_Agent
# ----------------------------------------------------------------------------------------------------------------------
folder_in = './data/ex_datasets/'
folder_out = './data/output/'
dct_config_agent = llm_config.get_config_azure()
# ----------------------------------------------------------------------------------------------------------------------
def ex_tools_test():
    A = [159.4, -4000.0, 300.0, 191.28, 275.45, 275.45, 229.54, 229.54, 80.0, 700.0, 159.4, 191.28, 700.0, 700.0, 300.0,80.0]
    #A = [-4000.0, 80.0, 80.0, 159.4, 159.4, 191.28, 191.28, 229.54, 229.54, 275.45, 275.45, 300.0, 300.0, 700.0, 700.0, 700.0],
    x = llm_tools.custom_func_sales_for_target_irr(A, 0.23)
    print(x)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_agent_IRR():
    q1 = "what is IRR of a cashflow below: -100, 30, 30, 30, 30"
    q2 = "what is required sale to achieve 0.23 IRR for cashflow below: 159.4, -4000.0, 300.0, 191.28, 275.45, 275.45, 229.54, 229.54, 80.0, 700.0, 159.4, 191.28, 700.0, 700.0, 300.0, 80.0"

    llm_cnfg = llm_config.get_config_openAI()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    tools = llm_tools.get_tool_IRR()
    A = llm_Agent.Agent(LLM, tools,verbose=False)
    llm_interaction.interaction_offline(A, [q1,q2], do_debug=False, do_spinner=True)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_agent_pandas(df):
    query0 = 'count number of records in dataframe'
    query1 = 'what is IRR of a cashflow below: -100, 30, 30, 30, 30'
    query2 = "How many records are available for investmentID F2-ABCTech?"
    #query3 = "List first 5 records as pandas dataframe for investmentID F2-ABCTech"
    query4 = 'calculate IRR of investmentID F2-ABCTech, assuming cashflow metricName is available in Csh.'
    query5 = 'Use available tool to evaluate required cashflow to generate for investmentID F2-ABCTech target IRR of 0.23, assuming cashflow data is encoded by metricName Csh'

    q = [query0,query1,query2,query4,query5]

    llm_cnfg = llm_config.get_config_openAI()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    tools = llm_tools.get_tools_pandas(df)
    tools.extend(llm_tools.get_tool_IRR())
    tools.extend(llm_tools.get_tool_sale_for_target_IRR())
    A = llm_Agent.Agent(LLM, tools,verbose=True)
    llm_interaction.interaction_offline(A, q, do_spinner=False)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_live(df):
    llm_cnfg = llm_config.get_config_openAI()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    #A = create_pandas_dataframe_agent(LLM, df, verbose=True)
    tools = llm_tools.get_tools_pandas(df)
    tools.extend(llm_tools.get_tool_IRR())
    tools.extend(llm_tools.get_tool_sale_for_target_IRR())
    A = llm_Agent.Agent(LLM, tools, verbose=True)
    llm_interaction.interaction_live(A, method='run', do_spinner=False)
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    df = pd.read_csv('./data/output/023de3ace6b3451082f61a3949c27239.csv')
    #ex_agent_IRR()
    ex_agent_pandas(df)
