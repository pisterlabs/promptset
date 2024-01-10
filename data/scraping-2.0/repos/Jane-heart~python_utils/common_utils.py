import openai
import os
from dotenv import load_dotenv

OPENAI_API_KEY = "OPENAI_API_KEY"
SERPAPI_API_KEY = "SERPAPI_API_KEY"

# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    '''
    prompt: 对应的提示
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
    temperature: 温度系数
    '''
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # 模型输出的温度系数，控制输出的随机程度
    )
    # 调用 OpenAI 的 ChatCompletion 接口
    return response.choices[0].message["content"]


"""
1. 安装dotenv库：在命令行中输入`pip install python-dotenv`。

2. 在Python脚本中导入dotenv库：`from dotenv import load_dotenv`

3. 调用load_dotenv函数，传入.env文件的路径作为参数：`load_dotenv('.env')`

4. 环境变量已经被导入到系统中，可以通过`os.environ`来访问它们。
使用示例：

"""
def import_env_variables(env_variables, path=".env"):
    """
    导入指定路径下的.env文件中的环境变量，并判断是否已经被导入
    :param env_variables: 要导入的环境变量名列表
    :param path: .env文件的路径，默认为当前路径下的.env文件
    ```python
    import_env_variables(["ENV_VAR1", "ENV_VAR2", "ENV_VAR3"])
    ```
    """
    load_dotenv(path)  # 加载.env文件中的环境变量

    for env_var in env_variables:
        if env_var in os.environ:
            print(f"变量名：{env_var}，已经被导入")
        else:
            if env_var in os.environ:
                print(f"变量名：{env_var}，正在被导入")
            else:
                print(f"变量名：{env_var}，不存在")