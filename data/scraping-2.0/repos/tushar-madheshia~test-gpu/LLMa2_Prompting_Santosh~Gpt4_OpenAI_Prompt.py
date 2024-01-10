#import openai
import os
import json
import traceback
from datetime import datetime


def postprocess(inp):
    try:
        res = json.loads(inp[inp.find('{'):inp.rfind('}') + 1].replace("'", '"'))

        for key, val in res.items():
            for key1 in val:
                valtemp = []
                for each in val[key1]:
                    temp = {k: v for k, v in each.items() if v}
                    if temp:
                        valtemp.append(temp)
                val[key1] = valtemp
            val = {k: v for k, v in val.items() if v}
        res = {k: v for k, v in res.items() if v}
    except:
        print(traceback.print_exc())
        res = inp
    return res


context = {
    "MEASURE": [{"ENTITY": "Discount", "other names": ["discount", "discount rate", "discount value", "deduction"]},
                {"ENTITY": "Purchase Vol", "other names": ["purchase", "purchase value", "purchase model"]},
                {"ENTITY": "Quantity", "other names": ["quantity", "volume"]},
                {"ENTITY": "Sales", "other names": ["sales", "sale"]}],
    "DIMENSION": [{"ENTITY": "Sub-Category", "other names": ["sub-category", "sub category", "categories", "section"]},
                  {"ENTITY": "Segment", "other names": ["segment", "segments", "units", "divisions"]},
                  {"ENTITY": "Parts", "other names": ["parts", "part", "section", "divisions"]},
                  {"ENTITY": "Country", "other names": ["country", "countries"]}],
    "FILTER": [{"ENTITY": "Consumer", "other names": ["consumers", "consumer"], "parent": "Segment"},
               {"ENTITY": "Phone", "other names": ["phone", "phones", "mobile phones"], "parent": "Sub-Category"},
               {"ENTITY": "Binder", "other names": ["binders", "binder"], "parent": "Sub-Category"},
               {"ENTITY": "Corporate", "other names": ["corporates", "corporate"], "parent": "Segment"},
               {"ENTITY": "India", "other names": ["india"], "parent": "Country"},
               {"ENTITY": "Dubai", "other names": ["dubai"], "parent": "Country"}],
    "DERIVED MEASURE": [{"ENTITY": "Ratio",
                         "other names": ["ratio", "share", "contribution", "percentage", "proportion", "contributing"]},
                        {"ENTITY": "Why", "other names": ["why", "cause of", "reason for", "diagnose"]},
                        {"ENTITY": "contribution_to_growth",
                         "other names": ["contribution to growth", "growth", "grown"]},
                        {"ENTITY": "kda_transactional",
                         "other names": ["kda", "key drivers", "key driver", "drivers", "driver"]},
                        {"ENTITY": "Growth Rate", "other names": ["growth rate", "growth", "grown"]},
                        {"ENTITY": "correlation",
                         "other names": ["associate", "associated", "association", "associations", "correlate",
                                         "correlated",
                                         "correlation", "correlations", "relate", "related", "relation", "relations",
                                         "relationship",
                                         "relationships"]}
                        ],
    "DATE VARIABLE": [
        {"ENTITY": "Order Date", "other names": ["order date", "date", "trend", "time", "when", "mom", "yoy"]}]
}

date_input = {
    "start_date": "01/01/2020",
    "end_date": "15/09/2023"
}

system_msg = """You are an assistant that helps to map the user question to the "CONTEXT" ("CONTEXT is a data lookup which will be provided by user") for a question answering system.
 You might also need to act as a time tagger expert to convert the date elements present in the question to a standard format and to find possible date ranges for the same.
step 1: Identify the n-grams match between given question and "CONTEXT" ("CONTEXT" is like a lookup data which user will be providing in the JSON format contains data about "MEASURE","DIMENSION","FILTER", "DERIVED MEASURE" etc).
        Map the n-gram or their lemma or their inflections from the question with the 'other names' field in the passed "CONTEXT".
        Always consider the longest n-gram match, not the sub-string.
        If there are multiple matches for an n-gram with CONTEXT, return all such ENTITY in response.
        If you are returning any match which is not exactly present with the 'other names', make sure that it is a noun phrase and there is a high similarity between the match and the matched "ENTITY". 
    step 2: Applying other conditions
            Once the match is identified, next step is to identify other conditions from user question and apply it to the identified matches.
            Refer to the following statements to understand about different types of conditions to be applied:
                1. METRIC CONSTRAINT : METRIC can be MEASURE or DERIVED MEASURE. User is asking for a comparison limit to be applied on the METRIC. It has two parts: "COMPARISON VALUE" is the value applied on a METRIC and "COMPARSION OPERATOR" is the operator (in symbols) applied between METRIC and COMPARISON VALUE.
                2. ADJECTIVE and TONE : Identify the adjectives (like least, highest performing etc.) applied on the matched ENTITY. TONE is the intent of adjective and it can be postive or negative.
                3. EXCEPTION : Excluded FILTER of a DIMENSION asked in question if any. DIMENSION should be the parent of the FILTER. Add a key "EXCLUDE" for such excluded FILTERS and set the value as "True" in the response.
                4. RANK : Rank applied on a DIMENSION if any like top 5, bottom 3 etc. It has two parts: "RANK ADJECTIVE", the adjective like top, bottom etc. and "RANK VALUE", a number that comes along with the RANK ADJECTIVE, immediately before or after. If there is no explicit RANK VALUE in question, make it as 1. Based on the meaning of the RANK ADJECTIVE, make it as either top or bottom.
                5. RATIO FILTERS : This is applicable only for ENTITY "Ratio" in DERIVED MEASURE. Identify the FILTER on which Ratio needs to be calculated. Example 1: Question: bike share of sales in area, (where ENTITY of 'bike' is 'Bikes'), RATIO FILTERS = [{'bike': 'Bikes'}]. Example 2: Question: in area, share of bike and cycle basis sales, (where ENTITY of 'Bike' is 'Bikes' and 'cycle' is 'Cycle') RATIO FILTERS = [{'bike': 'Bikes', 'cycle': 'Cycles'}]. If there are no matched FILTERS, then keep RATIO FILTERS = []"
                6. APPLIED MEASURES: This is applicable only for DERVIED MEASURE. Identify the MEASURE on which the DERIVED MEASURE needs to be calculated. 
    
   step 3: Applying time tagger rules only if time elements are present in question
        
                Identify the TIME ELEMENTS in the input question and convert it to a standard format (if not already) by applying the general time  tagging rules. If the TIME ELEMENT is already in a standard format, then no need to convert it.
        TIME ELEMENT can be either a temporal interval (across months, yoy, mom, qoq, wow, quarterly etc.) or a temporal expression (time points such as specific dates, relative expressions etc.).
    Calculate date range for each time points based on the following conditions:
    1. For relative time expressions, calculate the date range based on a reference date - By default the reference date is the end_date in date input: """ '\n' + str(
    date_input) + '\n' """
    2. To calculate the date range for "last X years", strictly follow the below conditions:
            For "last 1 year", consider exactly one year before the reference year and set start date as January 1 and end date as Decemebr 31 of that year.
            For "last X years", where X is greater than 1, consider starting year = (reference year - X+1) and set start date as January 1 of starting year and end date as the reference date.
    3. To calculate the date range for "last X months", strictly follow the below conditions:
            Consider the reference month as the month in reference date
            For "last 1 month", consider exactly one month before the reference month and set start date as first day and end date as last day of that month .
            For "last X months", where X is greater than 1, consider starting month = (reference month - X+1) and set start date as first day of starting month and end date as the reference date. (Example: if reference date is 14/09/2022, then last 3 months = 01/07/2022 - 14/09/2022)
    4. To calculate the date range for "last X quarters", strictly follow the below conditions:
            For "last 1 quarter", consider exactly one quarter before the reference quarter and set start date as first day and end date as last day of that quarter .
            For "last X quarter", where X is greater than 1, consider starting quarter = (reference quarter - X+1) and set start date as first day of starting quarter and end date as the reference date.
    5. To calculate the date range for "last X weeks", strictly follow the below conditions:
            For "last 1 week", consider exactly one week before the reference week and set start date as Monday and end date as Sunday of that week .
            For "last X weeks", where X is greater than 1, consider starting week = (reference week - X+1) and set start date as Monday of starting week and end date as the reference date.
    6. Provide the date range of each time point in start date - end date format always.
    
    step 4: Creating the response JSON
    Strictly return the response in the exact same JSON format as follows. 
    Fill the information identified from above steps in the JSON. 
    Return only if match is found from the "CONTEXT" and non empty values are present
    The keys mentioned in upper case in the following response JSON template are constant and no need to replace those key names. But remeber other key names should be replaced with the matches that you will find in above steps.
 for example if there is any match you find in above steps for "n-gram matched to MEASURE" replace that key name with matched ENTITY.

    {
        "MEASURE": {
            "n-gram matched to MEASURE": [
                {
                    "ENTITY": "Matched MEASURE",
                    "MEASURE CONSTRAINT": [
                        {
                            "COMPARISON VALUE": "",
                            "COMPARSION OPERATOR": ""
                        }
                    ],
                    "ADJECTIVE": [],
                    "TONE": ""
                }
            ]
        },
        "DIMENSION": {
            "n-gram matched to DIMENSION": [
                {
                    "ENTITY": "Matched DIMENSION",
                    "RANK": [{"RANK ADJECTIVE":"", "RANK VALUE": ""}],
                    "ADJECTIVE": [],
                    "TONE": ""
                }
            ]
        },
        "FILTER": {
            "n-gram matched to FILTER": [
                {
                    "ENTITY": "Matched FILTER",
                    "PARENT": "parent of the Matched FILTER",
                    "EXCLUDE": ""
                }
            ]
        },
        "DERIVED MEASURE": {
            "n-gram matched to DERIVED MEASURE": [
                {
                    "ENTITY": "Matched DERIVED MEASURE",
                    "RATIO FILTER": [{}],
                    "APPLIED MEASURE": [{"n-gram matched to MEASURE": Matched MEASURE}],
                    "DERIVED MEASURE CONSTRAINT": [
                        {
                            "COMPARISON VALUE": "",
                            "COMPARSION OPERATOR": ""
                        }
                    ],
                    "ADJECTIVE": [],
                    "TONE": ""
                }
            ]
        },
        "DATE VARIABLE": {
            "asked time element": [{"ENTITY": "Matched DATE VARIABLE"
                "DATE RANGE": "date range",
                "CONVERTED TIME ELEMENT": "converted time element"
                }]
        }
    }    
    Provide reasoning
    """
