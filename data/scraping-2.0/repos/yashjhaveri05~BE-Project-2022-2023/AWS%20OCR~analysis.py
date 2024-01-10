import pandas as pd
import numpy as np
import json
import openai

def toString(s): 
    string = "" 
    for element in s:
        string += element.capitalize() 
        if (element != s[len(s) - 1]):
          string += ", "
    return string

# report = pd.read_csv('output.csv')
# report = report.fillna(-1)
# columns = report.columns
# anomalies = {}

# file1 = open("output.txt","a")

# for i in range(len(report)):
#     parameter = report.loc[i, columns[0]]
#     observed = report.loc[i, columns[1]]
#     limits = report.loc[i, columns[3]]
#     vals = []
#     if int(observed) == -1 and int(limits) == -1:
#         pass
#     else:
#         limits = limits.split("-")
#         max_limit = float(limits[1])
#         min_limit = float(limits[0])
#         adj_max = float(max_limit/0.9)
#         adj_min = float(((min_limit/adj_max) - 0.1)*adj_max)
#         if float(observed) < adj_min:
#             vals.append("Low")
#             vals.append(1)
#         elif float(observed) > adj_max:
#             vals.append("High")
#             vals.append(1)
#         elif float(observed) < adj_max and float(observed) > max_limit:
#             vals.append("High")
#             vals.append(0)
#         elif float(observed) > adj_min and float(observed) < min_limit:
#             vals.append("Low")
#             vals.append(0)
#     if vals != []:
#         anomalies[parameter] = vals

# print(anomalies)

f = open('../Report Analysis/analysis.json')
report_list = json.load(f)
f.close()
g = open('../Report Analysis/priority.json')
priority_list = json.load(g)
g.close()

output = {'cholestrol': ['high', 1], 'MPV (Mean Platelet Volume)': ['high', 0], 'Vitamin B12 level (Serum,CMIA)': ['low', 0]}
# output = dict((k.lower(), v) for k, v in output.items()) 

def getAnalysis(output, report_list, priority_list):
  result_list = list(output.keys())
  for i in result_list:
     temp = findString(i)
     output[temp] = output[i]
     del output[i]
  result_list = list(output.keys())
  final_dict = {}
  high_priority_dict = {}
  high_pri = 6
  for i in result_list:
    rep_list = report_list.get(i)
    if rep_list != None:
      priority = priority_list.get(i)
      rep_list['priority'] = priority['priority']
      final_dict[i] = rep_list
      if priority['priority'] < high_pri:
        high_pri = priority['priority']
        high_priority_dict.clear()
        high_priority_dict[i] = rep_list
      elif priority['priority'] == high_pri:
        high_priority_dict[i] = rep_list

  high = []
  low = []
  for i in result_list:
      if (output.get(i)[0] == "high"):
         high.append(i)
      else:
         low.append(i)
  if (high):
    print("High Values: ", toString(high))
  if (low): 
    print("Low Values: ", toString(low))
  temp = output.get(list(high_priority_dict.keys())[0])[0]
  print("You should visit a", report_list.get(list(high_priority_dict.keys())[0])[temp][1], ", you have chances of", report_list.get(list(high_priority_dict.keys())[0])[temp][0])
  for i in list(high_priority_dict.keys()):
     generate(i, report_list, output.get(i)[0])
     del output[i]
  for i in list(output.keys()):
     generate(i, report_list, output.get(i)[0])
  # print("Final list of all: ", final_dict)
  # print("Highest priority: ", high_priority_dict)

def generate(elem, report_list, val):
  # print("\n", elem.upper())
  req_dict = report_list.get(elem)
  # print(req_dict["information"])
  # print(textGenerate("Write a note on " + elem))
  # print(textGenerate("Ill effects of having " + val + " values of " + elem))
  temp = req_dict["remedy_"+val]
  # print("Home remedies to be taken are: ", toString(temp))
  # print(textGenerate("Remedies that can be taken to cure "+ val + " values of " + elem))
  fin = {}
  fin["elem"] = elem
  fin["intro1"] = req_dict["information"]
  fin["intro2"] = textGenerate("Write a note on " + elem)
  fin["effects"] = textGenerate("Ill effects of having " + val + " values of " + elem)
  fin["rem1"] = "Home remedies to be taken are: " + toString(temp)
  fin["rem2"] = textGenerate("Remedies that can be taken to cure "+ val + " values of " + elem)
  print(fin)

def textGenerate(prompt):
  openai.api_key = "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"
  model_engine = "text-davinci-002"
  completion = openai.Completion.create(
    engine = model_engine,
    prompt = prompt,
    max_tokens = 1024,
    n=1,
    stop = None,
    temperature = 0.9,
  )
  response = completion.choices[0].text.lstrip()
  return response

# getAnalysis(output, report_list, priority_list)

# def getPriorityValue(word):
#   for key in priority_list:
#     if word == key:
#       return priority_list.get(key).get('priority')
#   for i in priority_list.keys():
#     if type(priority_list.get(i)) == list:
#       sub_list = priority_list.get(i)[0]
#       for key in sub_list:
#         if word == key:
#           return sub_list.get(key).get('priority')
#   return 6

# def getPriority(output):
#   key_dict = list(output.keys())
#   final_list = []
#   final_priority = 5
#   for i in key_dict:
#     val = getPriorityValue(i)
#     print(final_list)
#     if val < final_priority:
#       final_priority = val
#       final_list.clear()
#       final_list.append(i)
#     elif val == final_priority:
#       final_list.append(i)
#   return final_list

# def getAnalysis(result_list, output):
#   for i in result_list:
#     rep_list = report_list.get(i)
#     if rep_list == None:
#       for j in report_list.keys():
#         if type(report_list.get(j)) == list:
#           sub_list = report_list.get(j)[0]
#           for key in sub_list:
#             if i == key:
#               rep_list = sub_list.get(key)
#     print(rep_list)
#     output_value = output.get(i)
#     output_list = rep_list.get(output_value[0])
#     output_remedy = rep_list.get("remedy_" + output_value[0])
#     if output_value[1] == 1:
#       print("you need to visit a " + output_list[1])
#     if output_list[0]:
#       print(output_list[0])
#     print("Remedy: ", output_remedy)

# result_list = getPriority(output)
# getAnalysis(result_list, output)

def longestCommonSubstring(X, Y, m, n): 
  LongestCommonArray = [[0 for k in range(n+1)] for l in range(m+1)]  
  result = 0
  for i in range(m + 1):
      for j in range(n + 1):
          if (i == 0 or j == 0):
              LongestCommonArray[i][j] = 0
          elif (X[i-1] == Y[j-1]):
              LongestCommonArray[i][j] = LongestCommonArray[i-1][j-1] + 1
              result = max(result, LongestCommonArray[i][j])
          else:
              LongestCommonArray[i][j] = 0
  return result

def findString(Y):
  report_list_keys = report_list.keys()
  res = 0
  res_string = ""
  for i in report_list_keys:
    m = len(i)
    n = len(Y)
    temp = longestCommonSubstring(i, Y, m, n)
    if (res < temp):
      res = temp
      res_string = i
  return res_string
  
findString("lymphocytes")
getAnalysis(output, report_list, priority_list)
