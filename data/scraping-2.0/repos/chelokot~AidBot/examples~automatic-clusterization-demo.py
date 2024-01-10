# AidBot - Telegram bot project for finding volunteer help using semantic search
# Copyright (C) 2023
# Anastasia Mayorova aka EternityRei  <anastasiamayorova2003@gmail.com>
#    Andrey Vlasenko aka    chelokot   <andrey.vlasenko.work@gmail.com>

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# import openai
# from utils import config

# openai.api_key = config.api_key

# def get_embedding(text, model="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     embeddings_json = openai.Embedding.create(input=[text], model=model)['data']
#     return embeddings_json[0]['embedding']
    
# import requests
# from bs4 import BeautifulSoup as bs
# import json

# has_more = True
# skip_value = 0
# all_requests = []
# all_embeddings = []
# all_locations = []
# while has_more:
#     url = f'https://uahelpers.com/api/requests/search?location=&category=&skip={skip_value}'
#     response = requests.request("GET", url)

#     parsed_text = bs(response.text, 'lxml')
#     full_json_str = json.loads(parsed_text.find('p').text)

#     has_more = full_json_str['hasMore']
#     result = full_json_str['result']

#     for proposition in result:
#         values_list = []
#         proposition_json = json.dumps(proposition)
#         dict_json = json.loads(proposition_json)
#         for tag in dict_json:
#             values = dict_json[tag]
#             if type(values) == list:
#                 values_list.append(", ".join(values))
#             else:
#                 values_list.append(values)
#         if len(values_list) == 7:
#             values_list.append("null")
#         desc = dict_json['description']
#         name = dict_json['name']
#         location = dict_json['location']
#         if(len(desc) + len(name) > 25):
#           emb  = get_embedding(GoogleTranslator(source='auto', target='en').translate(name + ' ' + desc))
#           all_requests.append(name + ' ' + desc)
#           all_embeddings.append(emb)
#           all_locations.append(location)
#         else:
#           print(desc,len(desc))

#     skip_value += len(result)
#     print(len(all_requests))
    
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# data = all_embeddings

# linkage_data = linkage(data, method='ward', metric='euclidean')

# # Set the threshold distance to form clusters
# threshold = 0.8

# # Get the corresponding indexes of clusters for the vectors
# cluster_indexes = fcluster(linkage_data, t=threshold, criterion='distance')
# print(cluster_indexes)

# examples = {}

# for i in range(len(all_requests)):
#   if cluster_indexes[i] not in examples:
#     examples[cluster_indexes[i]] = [all_requests[i]]
#   else:
#     if(len(examples[cluster_indexes[i]]) < 8):
#       examples[cluster_indexes[i]].append(all_requests[i])

# for i in examples.keys():
#   print()
#   print(i)
#   for example in examples[i]:
#     print(example)
