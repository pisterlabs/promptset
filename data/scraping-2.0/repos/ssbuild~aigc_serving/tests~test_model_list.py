# coding=utf8
# @Time    : 2023/12/2 16:26
# @Author  : tk
# @FileName: test_model_list

import openai
openai.api_key = "112233"
openai.api_base = "http://106.12.147.243:9090/v1"

# # Test list models API
models = openai.Model.list()
print("Models:", models)
