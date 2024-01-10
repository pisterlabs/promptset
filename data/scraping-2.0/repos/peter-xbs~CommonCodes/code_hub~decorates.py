# _*_ coding:utf-8 _*_

"""
@Time: 2022/3/19 6:55 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
"""
import time


def calc_time(func):
    def _calc_time(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        ed = time.time()
        print("%s cost time:%s" % (getattr(func, "__name__"), ed - start))
        # l = list(args)
        # print("args: %s" % args)
        return ret

    return _calc_time


### LOG装饰器
# 大家调用前将本cell粘贴到自己 notebook中运行即可
# 大家调用前补充PROJ & USAGE & USER 信息，可自动记录到Log中，方便后续和标注公司对账
from datetime import datetime

PROJ = """测试"""  ## 添加项目名称
USAGE = """测试"""  ## 添加用途目的
USER = """XX"""  ## 添加使用人员
LOG_PATH = 'xx'


def record_time(func):
    # 日志目录在./logs/ 下面，每天生成一个日志，记录调用时间和调用次数
    log_file = f'{LOG_PATH}/{datetime.today().strftime("%Y-%m-%d")}.log'

    def wrapper(*args, **kwargs):
        start = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        result = func(*args, **kwargs)
        end = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        res = result[0].replace("\n", "")
        args_ = str(args)
        with open(log_file, 'a') as f:
            f.write(
                f"{PROJ}_{USAGE}_{USER}_{func.__name__},prompt: {args_}res:{res},start:{start},end:{end},counts:{str(result[1])} \n")
        return result[0]

    return wrapper


@record_time
def chatgpt_azure(prompt, temperature=0, top_p=0.95):
    import openai
    import traceback
    # 添加配置信息
    openai.api_type = "azure"
    openai.api_base = "xx"  # 请填写您的：终结点（Endpoint）
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "xx"  # 请填写您的：API密钥（API Key）
    model_name = "gpt-35-turbo-cmri"  # 请填写您的：模型名称

    rsp = ''
    cnt = 0
    while (not rsp and cnt < 2):
        try:
            rsp = openai.ChatCompletion.create(
                engine=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            err = traceback.format_exc()
            print(err)
            rsp = ''
            print('request error, sleep {} seconds'.format(10))
            time.sleep(2)
        cnt += 1

    try:
        res = rsp['choices'][0]['message']['content']
        if not isinstance(res, str):
            return '', cnt
        return res, cnt
    except Exception as e:
        err = traceback.format_exc()
        print(err)
    return '', cnt