# Implement demo code by Streamlit: https://github.com/streamlit/llm-examples/tree/main

import streamlit as st
import pandas as pd
import openai
import types
import inspect
import textwrap
import json


#%%###################################################################################################################### 
# GPT AGENT CONFIGURATION
#########################################################################################################################

def chat_gpt_basic_BI_agent (sentence):

    # Configure
    system_message = """        
        You are a Business Intelligence (BI) analyst that produces reports to answer business questions and meet information requests from "business users". 
        
        Your job is to: 
            1. interpret a business question or information request from a "business user".
            2. choose the correct type of report and identify the dimensions, filters, and metrics to be used in that report. Then call a function that will generate and send that report to the business user. Then respond 'Report sent' to the user.
            3. if not possible to meet the requirements, then respond to the user: "I cannot deliver that information with the data and reports that I have available" and add a brief explanation of the reason.

        No other response to the user is acceptable: you must either act and respond as indicated in item 2 or respond as described in item 3. 

        The type of reports that you have available are:
            a) Generic Spend Report: Flexible report that you can use to answer many user questions about historical transaction volumes and spend. You can specify various filters, dimensions, and metrics for this report.
            b) Benchmarking Spend Report: Answers questions such as: how are my vendors performing relative to peers? Use this report to compare the rates of vendors against competitors that perform similar work. Dimensions and metrics are fixed, but you may specify filters.
            c) Market Share Report: Answers questions such as: what are the top N vendors in a given category? Use this report to show the largest vendors available in the market and their share. You may specify one metric ('volume_share' or 'spend_share'), a vendor count limit (top N), and filters.
            d) Vendor Market Share Map: It displays a map to answer questions such as: what markets does a given vendor serve and what share does it have? who is the market leader for a given category? and where does that market leader offer services? You may specify one metric ('volume share' or 'spend share') and filters.

        The data that these reports can access is a table of hospital expenditures in a PostgreSQL database. The table has records for every purchase or service transaction and contains the following fields:
            - transaction_datetime : the date and time of the purchase or service (type: timestamp)
            - spend_amount         : the cost of the purchase or service in US$ (type: numeric)
            - vendor_name          : the name of the vendor company (a.k.a. service provider, manufacturer, distributor, contractor) that provided the product or service (type: char)
            - category_name        : the category of the expense according to a standard classification system (type: char)
            - hospital_name        : the name of the hospital (also refered to as "facility") that incurred the expense (type: char)
            - division_name        : the name of the division of the hopital or hospital group that incurred the expense (type: char)
            - department_name      : the name of the department of the hopital that incurred the expense (type: char)
            - gl_account_name      : the General Ledger account of the expenditure in the accouting system of the hospital or hospital group (type: char)
            - geographic_market    : the name of the geographical region where the hospital facility is located under a standard classification system (imagine MSAs, CBSAs, CSAs, DMAs, or similar systems.) (type: char)

        All reports will have the following "filters" that you can specify in order to include only the necesary expenditures in each report:  
            - vendor_filter_list: List of vendor company names to be included
            - category_filter_list: List of spend category names to be included 
            - hospital_filter_list: List of hospital names (a.k.a. facility names) to be included 
            - department_filter_list: List of department names to be included
            - division_filter_list: List of division names to be included  
            - gl_account_filter_list: List of GL account names to be included
            - time_period_filter_list: List of "period names" to be included. Each period will be calculated by the report based on transaction_date and the current date of the report. The valid period names that you may specify are: 
                'LAST YEAR', 'LAST QUARTER', 'LAST MONTH', 'LAST WEEK';
                'LAST N YEARS', 'LAST N QUARTERS', 'LAST N MONTHS', 'LAST N WEEKS', 'LAST N DAYS' (where N = integer you must specify);
                'YEAR TO DATE', 'QUARTER TO DATE', 'MONTH TO DATE'; 
                'YEAR yyyy', 'QUARTER yyyy-q', 'MONTH yyyy-mm' (where you can specify an exact year "yyyy", quarter number "q", and month number "mm"); 
                'FROM yyyy-mm-dd TO yyy-mm-dd' (where you can specify the start and end-date of an ad-hoc period in YYYY-MM-DD format);
                'STLY' (means "same time last year" and its calculated taking the previous period in the list and shifting it 1 year back in time);
                and 'ALL TIME' (if the report must include all transaction dates, e.g. not have any "time filter")

        Some reports allow you to specify dimensions. Please specify any metric names that you need. Valid dimension names include: 
            'vendor_name', 'category_name', 'hospital_name', 'division_name', 'department_name', 'gl_account_name'; 
            'transaction_year', 'transaction_quarter', 'transaction_month', 'transaction_day', and 'transaction_week' (which come from grouping transaction_datetime). 

        Some reports allow you to specify metrics. Please specify any metric names that you need. Valid metrics name and their definitions are:
            'total_volume' = count of unique transactions,
            'total_amount' = sum(spend_amount),
            'min_amount'   = min(spend_amount),
            'max_amount'   = max(spend_amount),
            'avg_amount'   = total_amount / total_volume,
            'pXX_amount'   = percentile of spend_amount (where XX = integer you specify, for example 'p50_amount' is the median),
            'volume_share' = total_volume of a vendor as a percent of the grand total volume in the "market" (defined as the geographic_market and the category filters in the report), and              
            'spend_share'  = total_amount of a vendor as a percent of the grand total amount in the "market" (defined as the geographic_market and the category filters in the report).                

        Additional instructions:
        - Do NOT explain your thought process or your analytics methodology in your final response to the business user. The business user just wants the report (which you can deliver by calling the right function) or a simple explanation of why you cannot provide it.
        - Do NOT include technical details, python code, javascript code, or SQL queries in your final response to the business user. Business users don't understand those languages and cannot access any of the reporting functions themselves. 
        - Keep each report as simple as possible: Do not add any filters, dimensions, or metrics that are not essential to answer the business question.  
        """

    prompt = f"""
        Please answer the following question or information request from a business user:
        [SENTENCE]
        """

    # Cleanup
    prompt = textwrap.dedent(prompt).strip()
    prompt = prompt.replace('[SENTENCE]', sentence)
    if system_message:
        system_message = textwrap.dedent(system_message).strip()

    # Return
    return prompt, system_message


# DEFINE THE CALLBACK FUNCTION SPECS FOR CHAT GPT
functions_definitions = [
    # generate_and_send_generic_spend_report
    {
        "name": "generate_and_send_generic_spend_report",
        "description": "Use this function to generate a Generic Spend Report and send it to the business user. ",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string", 
                    "description": "The name or title of the report to be displayed to the business user",
                },
                "vendor_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of vendor companies that the report should include. If this filter is needed, please pass a list of names for one or more vendors. If this optional parameter is not provided, then all vendors will be used.",
                },
                "category_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of spend categories that the report should include. If this filter is needed, please pass a list of names for one or more spend categories. If this optional parameter is not provided, then all categories will be used.",
                },
                "hospital_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of hospitals that the report should include. If this filter is needed, please pass a list of names for one or more hospitals/facilities. If this optional parameter is not provided, then all hospitals will be used.",
                },
                "department_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of departments that the report should include. If this filter is needed, please pass a list of names for one or more departments. If this optional parameter is not provided, then all departments will be used.",
                },
                "division_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of divisions that the report should include. If this filter is needed, please pass a list of names for one or more divisions. If this optional parameter is not provided, then all divisions will be used.",
                },
                "gl_account_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of GL accounts that the report should include. If this filter is needed, please pass a list of names for one or more GL accounts. If this optional parameter is not provided, then all GL Accounts will be used.",
                },
                "time_period_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of time periods that the report should include. If this filter is needed, please pass a list of one or more period names such as: ['YEAR TO DATE', 'STLY', 'LAST 5 YEARS']. For the complete list of acceptable period names, please refer to the system instructions provided earlier. If this optional parameter is not provided, the report will either apply a default period or include all time.",
                },
                "fiscal_calendar": {
                    "type": "boolean",
                    "description": "If True, the time periods in the filters and dimensions will be calculated using fiscal years and fiscal quarters. If False, they will be calculated using normal calendar years and quarters. If this optional parameter is not provided, then normal calendar periods will be used as default.",
                },
                "metrics": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of metric names that the report should display. Example: ['total_volume', 'p50_amount', 'spend_share']. For the complete list of acceptable metric names, please refer to the system instructions provided earlier. This is an optional parameter: don't supply it if the business question can be answered without any metrics.",
                },
                "dimensions": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of dimension names that the report should display. Example: ['vendor_name', 'transaction_year']. For the complete list of acceptable dimention names, please refer to the system instructions provided earlier. This is an optional parameter: don't supply it if dimensions aren't strictly necesary to answer the business question.",
                },
                "user_request": {
                    "type": "string",
                    "description": "The text of the business user request or question that you are seeking to answer with this report. It will not be used in producing the report (it's passed only for documentation purposes.)",
                },
            },
            "required": ["title", "user_request"],
        },
    },
    # generate_and_send_benchmarking_spend_report
    {
        "name": "generate_and_send_benchmarking_spend_report",
        "description": "Use this function to generate a Benchmarking Spend Report and send it to the business user. ",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The name or title of the report to be displayed to the business user",
                },
                "vendor_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of vendor companies that the report should include. If this filter is needed, please pass a list of names for one or more vendors. If this optional parameter is not provided, then all vendors will be used.",
                },
                "category_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of spend categories that the report should include. If this filter is needed, please pass a list of names for one or more spend categories. If this optional parameter is not provided, then all categories will be used.",
                },
                "hospital_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of hospitals that the report should include. If this filter is needed, please pass a list of names for one or more hospitals/facilities. If this optional parameter is not provided, then all hospitals will be used.",
                },
                "department_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of departments that the report should include. If this filter is needed, please pass a list of names for one or more departments. If this optional parameter is not provided, then all departments will be used.",
                },
                "division_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of divisions that the report should include. If this filter is needed, please pass a list of names for one or more divisions. If this optional parameter is not provided, then all divisions will be used.",
                },
                "gl_account_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of GL accounts that the report should include. If this filter is needed, please pass a list of names for one or more GL accounts. If this optional parameter is not provided, then all GL Accounts will be used.",
                },
                "time_period_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of time periods that the report should include. If this filter is needed, please pass a list of one or more period names such as: ['YEAR TO DATE', 'STLY', 'LAST 5 YEARS']. For the complete list of acceptable period names, please refer to the system instructions provided earlier. If this optional parameter is not provided, the report will either apply a default period or include all time.",
                },
                "fiscal_calendar": {
                    "type": "boolean",
                    "description": "If True, the time periods in the filters and dimensions will be calculated using fiscal years and fiscal quarters. If False, they will be calculated using normal calendar years and quarters. If this optional parameter is not provided, then normal calendar periods will be used as default.",
                },
                "user_request": {
                    "type": "string",
                    "description": "The text of the business user request or question that you are seeking to answer with this report. It will not be used in producing the report (it's passed only for documentation purposes.)",
                },
            },
            "required": ["title", "user_request"],
        },
    },
    # generate_and_send_market_share_report
    {
        "name": "generate_and_send_market_share_report",
        "description": "Use this function to generate a Market Share Report and send it to the business user. ",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The name or title of the report to be displayed to the business user",
                },
                "share_metric": {
                    "type": "string",
                    "description": "Which share metric should the report display. Pick one of these two options: 'volume_share' or 'spend_share'.",
                },
                "vendor_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of vendor companies that the report should include. If this filter is needed, please pass a list of names for one or more vendors. If this optional parameter is not provided, then all vendors will be used.",
                },
                "category_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of spend categories that the report should include. If this filter is needed, please pass a list of names for one or more spend categories. If this optional parameter is not provided, then all categories will be used.",
                },
                "hospital_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of hospitals that the report should include. If this filter is needed, please pass a list of names for one or more hospitals/facilities. If this optional parameter is not provided, then all hospitals will be used.",
                },
                "department_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of departments that the report should include. If this filter is needed, please pass a list of names for one or more departments. If this optional parameter is not provided, then all departments will be used.",
                },
                "division_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of divisions that the report should include. If this filter is needed, please pass a list of names for one or more divisions. If this optional parameter is not provided, then all divisions will be used.",
                },
                "gl_account_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of GL accounts that the report should include. If this filter is needed, please pass a list of names for one or more GL accounts. If this optional parameter is not provided, then all GL Accounts will be used.",
                },
                "time_period_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of time periods that the report should include. If this filter is needed, please pass a list of one or more period names such as: ['YEAR TO DATE', 'STLY', 'LAST 5 YEARS']. For the complete list of acceptable period names, please refer to the system instructions provided earlier. If this optional parameter is not provided, the report will either apply a default period or include all time.",
                },
                "fiscal_calendar": {
                    "type": "boolean",
                    "description": "If True, the time periods in the filters and dimensions will be calculated using fiscal years and fiscal quarters. If False, they will be calculated using normal calendar years and quarters. If this optional parameter is not provided, then normal calendar periods will be used as default.",
                },
                "user_request": {
                    "type": "string",
                    "description": "The text of the business user request or question that you are seeking to answer with this report. It will not be used in producing the report (it's passed only for documentation purposes.)",
                },
            },
            "required": ["title", "share_metric", "user_request"],
        },
    },
    # generate_and_send_vendor_market_share_map
    {
        "name": "generate_and_send_vendor_market_share_map",
        "description": "Use this function to generate a Vendor Market Share Map and send it to the business user. ",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The name or title of the report to be displayed to the business user",
                },
                "share_metric": {
                    "type": "string",
                    "description": "Which share metric should the report display. Pick one of these two options: 'volume_share' or 'spend_share'.",
                },
                "vendor_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of vendor companies that the report should include. If this filter is needed, please pass a list of names for one or more vendors. If this optional parameter is not provided, then all vendors will be used.",
                },
                "category_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of spend categories that the report should include. If this filter is needed, please pass a list of names for one or more spend categories. If this optional parameter is not provided, then all categories will be used.",
                },
                "hospital_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of hospitals that the report should include. If this filter is needed, please pass a list of names for one or more hospitals/facilities. If this optional parameter is not provided, then all hospitals will be used.",
                },
                "department_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of departments that the report should include. If this filter is needed, please pass a list of names for one or more departments. If this optional parameter is not provided, then all departments will be used.",
                },
                "division_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of divisions that the report should include. If this filter is needed, please pass a list of names for one or more divisions. If this optional parameter is not provided, then all divisions will be used.",
                },
                "gl_account_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of GL accounts that the report should include. If this filter is needed, please pass a list of names for one or more GL accounts. If this optional parameter is not provided, then all GL Accounts will be used.",
                },
                "time_period_filter_list": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of time periods that the report should include. If this filter is needed, please pass a list of one or more period names such as: ['YEAR TO DATE', 'STLY', 'LAST 5 YEARS']. For the complete list of acceptable period names, please refer to the system instructions provided earlier. If this optional parameter is not provided, the report will either apply a default period or include all time.",
                },
                "fiscal_calendar": {
                    "type": "boolean",
                    "description": "If True, the time periods in the filters and dimensions will be calculated using fiscal years and fiscal quarters. If False, they will be calculated using normal calendar years and quarters. If this optional parameter is not provided, then normal calendar periods will be used as default.",
                },
                "user_request": {
                    "type": "string",
                    "description": "The text of the business user request or question that you are seeking to answer with this report. It will not be used in producing the report (it's passed only for documentation purposes.)",
                },
            },
            "required": ["title", "share_metric", "user_request"],
        },
    },
]



# Function definitions with global connection and logging function
def generate_and_send_generic_spend_report( title                   ,
                                            user_request            ,
                                            vendor_filter_list      = None, 
                                            category_filter_list    = None,
                                            hospital_filter_list    = None,
                                            department_filter_list  = None,
                                            division_filter_list    = None,
                                            gl_account_filter_list  = None,
                                            time_period_filter_list = None,
                                            fiscal_calendar         = None, 
                                            metrics                 = None, 
                                            dimensions              = None):
    command = f"""generate_and_send_generic_spend_report(
        title                  : {title                  } 
        vendor_filter_list     : {vendor_filter_list     } 
        category_filter_list   : {category_filter_list   } 
        hospital_filter_list   : {hospital_filter_list   } 
        department_filter_list : {department_filter_list } 
        division_filter_list   : {division_filter_list   } 
        gl_account_filter_list : {gl_account_filter_list } 
        time_period_filter_list: {time_period_filter_list} 
        fiscal_calendar        : {fiscal_calendar        } 
        metrics                : {metrics                } 
        dimensions             : {dimensions             } 
        user_request           : {user_request           } 
        )
    """
    response = f"Generic spend report has been generated. Please respond to the user: 'Report {title} sent.'"
    return response

def genete_and_send_benchmarking_spend_report(    title                   ,
                                                    user_request            , 
                                                    vendor_filter_list      = None, 
                                                    category_filter_list    = None,
                                                    hospital_filter_list    = None,
                                                    department_filter_list  = None,
                                                    division_filter_list    = None,
                                                    gl_account_filter_list  = None,
                                                    time_period_filter_list = None,
                                                    fiscal_calendar         = None):
    command = f"""generate_and_send_benchmarking_spend_report(
        title                  : {title                  }
        vendor_filter_list     : {vendor_filter_list     }
        category_filter_list   : {category_filter_list   }
        hospital_filter_list   : {hospital_filter_list   }
        department_filter_list : {department_filter_list }
        division_filter_list   : {division_filter_list   }
        gl_account_filter_list : {gl_account_filter_list }
        time_period_filter_list: {time_period_filter_list}
        fiscal_calendar        : {fiscal_calendar        }
        user_request           : {user_request           }
        )
    """
    response = f"Benchmarking report has been generated. Please respond to the user: 'Report {title} sent.'"
    return response

def generate_and_send_market_share_report(  title                   , 
                                            share_metric            ,
                                            user_request            ,
                                            vendor_filter_list      = None, 
                                            category_filter_list    = None,
                                            hospital_filter_list    = None,
                                            department_filter_list  = None,
                                            division_filter_list    = None,
                                            gl_account_filter_list  = None,
                                            time_period_filter_list = None,
                                            fiscal_calendar         = None):
    command = f"""generate_and_send_market_share_report(
        title                  : {title                  } 
        share_metric           : {share_metric           } 
        vendor_filter_list     : {vendor_filter_list     } 
        category_filter_list   : {category_filter_list   } 
        hospital_filter_list   : {hospital_filter_list   } 
        department_filter_list : {department_filter_list } 
        division_filter_list   : {division_filter_list   } 
        gl_account_filter_list : {gl_account_filter_list } 
        time_period_filter_list: {time_period_filter_list} 
        fiscal_calendar        : {fiscal_calendar        } 
        user_request           : {user_request           } 
        )
    """
    response = f"Marketshare report has been generated. Please respond to the user: 'Report {title} sent.'"
    return response

def generate_and_send_vendor_market_share_map(  title                   , 
                                                share_metric            ,
                                                user_request            ,
                                                vendor_filter_list      = None, 
                                                category_filter_list    = None,
                                                hospital_filter_list    = None,
                                                department_filter_list  = None,
                                                division_filter_list    = None,
                                                gl_account_filter_list  = None,
                                                time_period_filter_list = None,
                                                fiscal_calendar         = None):
    command = f"""generate_and_send_vendor_market_share_map(
        title                  : {title                  } 
        share_metric           : {share_metric           } 
        vendor_filter_list     : {vendor_filter_list     } 
        category_filter_list   : {category_filter_list   } 
        hospital_filter_list   : {hospital_filter_list   } 
        department_filter_list : {department_filter_list } 
        division_filter_list   : {division_filter_list   } 
        gl_account_filter_list : {gl_account_filter_list } 
        time_period_filter_list: {time_period_filter_list} 
        fiscal_calendar        : {fiscal_calendar        } 
        user_request           : {user_request           } 
        )
    """
    response = f"Marketshare Map has been generated. Please respond to the user: 'Report {title} sent.'"
    return response


# Map function names to actual callable functions available in all namespaces in the stack
callable_functions = {name: func
                        for frame_info in inspect.stack()
                        for name, func in frame_info.frame.f_globals.items()
                        if isinstance(func, types.FunctionType) and name in [f['name'] for f in functions_definitions] }

# Generate the list  of "tools" (only functions for now)
tool_definitions = []
for f in functions_definitions:
    if f['name'] in callable_functions.keys():
        tool_definitions.append({"type": "function", "function": f})



#%%###################################################################################################################### 
# ACTUAL APPLICATION
#########################################################################################################################

# Load settings
openai_api_key = st.secrets["openai"]["key"]
_, system_message = chat_gpt_basic_BI_agent('no prompt yet')

# Display app basic information
st.title("ChatGPT 4.0 Questions with Function Calling")
st.caption("ChatGPT is a report generation robot. Ask a business question about hospital expenditures. ChatGPT should interpret it and call the right reporting function.")

# Initialize the web session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": system_message}]

# Execute
for msg in st.session_state.messages:
    if msg["role"] != 'system':
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    # if not bi_analyst_pwd or bi_analyst_pwd != 'Valify2024':
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    openai.api_key = openai_api_key

    # Prepare the message
    # prompt, _ = chat_gpt_basic_BI_agent(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Call the API
    response = openai.ChatCompletion.create(model       = 'gpt-4', 
                                            messages    = st.session_state.messages, 
                                            tools       = tool_definitions,
                                            tool_choice = 'auto',
                                            temperature = 0.5, 
                                            )
    
    # Uppack the API response
    response_message = response.choices[0].message
    response_content = response_message.content
    st.session_state.messages.append(dict(response_message))
    response_attributes = { 
        'id':                response['id'],
        'response_ms':       response.response_ms,
        'model':             response['model'],
        'prompt_tokens':     response['usage']['prompt_tokens'],
        'completion_tokens': response['usage']['completion_tokens'],
        'total_tokens':      response['usage']['total_tokens'],
        }

    # Check if the response contains a function call
    if 'tool_calls' in response_message:

        response_content = []  # Turn blank string into a list

        # Execute all the tools called by ChatGPT
        for tool in response_message['tool_calls']:

            if tool['type']  == 'function':
                # Call the function
                # TODO: the JSON response may not always be valid; be sure to handle errors
                # TODO: Check when no longer in beta: client.beta.threads.runs.submit_tool_outputs
                function_name    = tool["function"]["name"]
                function_args    = tool["function"]["arguments"]
                function_args    = json.loads(function_args)
                function_to_call = callable_functions[function_name]
                function_response = function_to_call(**function_args)

                # Prepare the info on the function call and function response to GPT
                st.session_state.messages.append({  "role"         : "tool", 
                                                    "tool_call_id" : tool['id'], 
                                                    "name"         : function_name, 
                                                    "content"      : function_response,
                                                    })
                    
                # When the response is a call-back function, return the function results to the user
                response_content.append({   "function called" : function_name, 
                                            "arguments"       : function_args,
                                            "response"        : function_response,
                                            })

    # Display the response to the app user
    st.chat_message("assistant").write(response_content)

