import Levenshtein
import numpy as np

import pandas as pd
import plotly
import plotly.express as px
from dataclasses import dataclass
from typing import List, Dict, Set
import numpy
from Levenshtein import ratio
import itertools
import networkx
import openai
import time
import pickle
import swifter

aux_data = pd.read_csv("data/jobs-dump-6mo.csv")

#remap userids of nan to -1
aux_data.user_id = aux_data.user_id.fillna(-1).astype(int)
aux_data.create_time = pd.to_datetime(aux_data.create_time.fillna("1970-01-01 00:00:00.0"))

data = aux_data[["id","create_time","update_time","history_id","tool_id", "session_id", "user_id", "exit_code", "dynamic_tool_id" ]].copy()

#add error column to data... exit code is set and not 0
data["Error"] = ~pd.isna(data.exit_code) & data.exit_code != 0

#lets look into exit code
frame = data.loc[data.Error].exit_code
px.bar(frame.value_counts()).show()

#Ooh... there is 127 (command not found) in there... those are errors... Those need isolation
tools_with_127=data.loc[data.exit_code == 127].tool_id.unique()

#calculate if there _always is a 127 -> command not found, if its sometimes, then probably the tool developer doesn't know how to use error codex
subs = data.loc[data.tool_id.isin(tools_with_127)]
tools_with_127=subs.groupby(subs.tool_id).Error.mean()
tools_with_127 = tools_with_127.loc[tools_with_127 > .9]
#tools 127 need investigation.



#calculate invocation number (if the job submitted by the same user is within 3 seconds before the previous submission its the same invocation)
@dataclass
class Usertime:
    uid : int
    time: any
    invocation_id: int


user_times : Dict[int,Usertime] = {}
invocation_ctr : List[int] = []
invoc_nr : int = 0

data.set_index("create_time",inplace=True)
data.sort_index(inplace=True)

for idx,row in data.iterrows():
    if row.user_id not in user_times:
        ut = Usertime(row.user_id, idx, invoc_nr)
        user_times[row.user_id] = ut
        invoc_nr += 1

    ut = user_times[row.user_id]
    if idx-ut.time > pd.Timedelta("3sec"):
        ut.invocation_id = invoc_nr
        invoc_nr += 1

    invocation_ctr.append(ut.invocation_id)

data['invocation_id'] = invocation_ctr
#set the index back
data.reset_index(inplace=True)

#calculate if the invocation has an error in any of the children
invocation_error=data.groupby(data.invocation_id).Error.sum() > 0
invocation_error.name="invocation_error"
data = data.set_index("invocation_id").join(invocation_error).reset_index()

#calculate the percentage of failed invocations... for whatever reason
invocation_count = data.groupby("tool_id").invocation_id.nunique().sort_values()
erronious_invocation = data.loc[data.invocation_error].groupby("tool_id").invocation_id.nunique().sort_values()
failed_invocation_percentage = (erronious_invocation/invocation_count).fillna(0).sort_values()
failed_invocation_percentage.name = "invocation_error_rate"

#calculate users affected
failed_invocation_users = data.loc[data.invocation_error].groupby("tool_id").user_id.nunique().sort_values()


corr_frame = pd.DataFrame(failed_invocation_percentage)
corr_frame["invocation_count"] = invocation_count
corr_frame["users_affected"] = failed_invocation_users

#export failed invocations
corr_frame.to_csv("data/invocation_error.csv")


#define tools to investigate
tti=failed_invocation_percentage > .20
reasons_per_tool: Dict[str,Dict[str,Set[str]]] = {}

for tool in tti.loc[tti].index:
    tool_data=aux_data.loc[aux_data.tool_id == tool].copy()
    tool_data["Error"] = ~pd.isna(tool_data.exit_code) & tool_data.exit_code != 0
    tool_reasons = set(tool_data.loc[tool_data.Error].tool_stderr)
    ltool_reasons = list(tool_reasons)
    indexes_to_process = set(range(len(ltool_reasons)))
    classes : Dict[str,Set[str]] = {}

    while indexes_to_process:
        elem = indexes_to_process.pop()
        cl = set([elem])
        for j in indexes_to_process:
            if Levenshtein.ratio(ltool_reasons[elem],ltool_reasons[j]) > .9:
                cl.add(j)
        indexes_to_process.difference_update(cl)
        classes[ltool_reasons[elem]] = set([ltool_reasons[x] for x in cl])
    reasons_per_tool[tool] = classes
    print("processed_tool")

pickle.dump(reasons_per_tool,open("data/reasons_per_tool.pickle","wb"))

openai.api_key = "my-secret-key"
openai.Model.list()

openai_thoughts = {}
req_ctr = 0
req_ctr = 0

for k in reasons_per_tool:
    openai_thoughts[k] = {}
    for msg in reasons_per_tool[k]:
        if isinstance(msg,str):
            retry_count=0
            result = None
            while retry_count < 4:
                try:
                    req_ctr +=1
                    result = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user",
                           "content": """Is this likely an issue with the installation? Answer only with yes or no. %s"""%msg}])
                    break
                except openai.error.RateLimitError as e:
                    print("Hit rate limit, backing off for 60 seconds")
                    time.sleep(57)
                except (openai.error.InvalidRequestError, openai.error.InvalidAPIType, openai.error.AuthenticationError) as e:
                    result = e
                    break
                except Exception as e:
                    pass
            openai_thoughts[k][msg] = result
            time.sleep(3)
            if req_ctr % 10 == 0:
                pickle.dump(openai_thoughts,open("data/openai_thoughts.pickle","wb"))

pickle.dump(openai_thoughts,open("data/openai_thoughts.pickle","wb"))

responses = set()
for k in openai_thoughts:
    for msg in openai_thoughts[k]:
        openai_resp = openai_thoughts[k][msg]
        if isinstance(openai_resp, openai.openai_object.OpenAIObject):
            responses.add(openai_resp['choices'][0]["message"]["content"])

response_map = {}
for i in responses:
    if i.strip().lower().startswith("yes"):
        response_map[i] = "Yes"
    elif i.strip().lower().startswith("no"):
        response_map[i] = "No"
    else:
        response_map[i] = "Maybe"
