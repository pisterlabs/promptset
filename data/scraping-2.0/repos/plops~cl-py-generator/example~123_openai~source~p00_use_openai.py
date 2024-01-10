#|default_exp p00_use_openai
# python -m venv ~/bardapi_venv
# python -m venv ~/bardapi_env; . ~/bardapi_env/bin/activate; pip install openai toml
# 
# deactivate
# https://platform.openai.com/docs/api-reference/authentication
# https://github.com/openai/openai-python
# env.toml contains this:
# [keys]
# key_token = " ... " 
import os
import time
import toml
from openai import OpenAI
start_time=time.time()
debug=True
_code_git_version="17500fa0a338b604d7d351b5f3183c9aa029f414"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/123_openai/source/"
_code_generation_time="01:22:04 of Sunday, 2023-12-17 (GMT+1)"
# either set cookies here or read them from env.toml file
config_path="env.toml"
if ( os.path.exists(config_path) ):
    with open(config_path, "r") as f:
        data=toml.load(f)
    keys=data["keys"]
else:
    print("Warning: No env.toml file found.")
client=OpenAI(api_key=keys["key_token"])
chat_completion=client.chat.completions.create(messages=[dict(role="user", content="Say this is a test")], model="gpt-3.5-turbo")