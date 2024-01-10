#!/usr/bin/env python3
# encoding: utf-8
import threading
import vim
import os

# python async call example
# https://github.com/roxma/nvim-ascript/blob/abc89587d6d6c83eb28d62669f0111bb77c00d07/autoload/ascript.vim#L32

# 全局llm
llm = None

# 全局 LLM
CustomLLM = None
OpenAI = None

# 默认都不支持流式输出，目前只实现了 api2d 的流式输出
stream_output = False

def safe_vim_eval(expression):
    try:
        return vim.eval(expression)
    except vim.error:
        return None

def loading_done():
    vim.command("call nvim_ai#loading#done()")

def import_deps():
    global CustomLLM, OpenAI
    from .langchain_llm import CustomLLM
    from .langchain_llm import OpenAI
    vim.async_call(loading_done)
    pass

def import_deps_async():
    vim.command("call nvim_ai#loading#start('ChatGPT 初始化中...')")
    fetch_thread = threading.Thread(target=import_deps, args=())
    fetch_thread.daemon = True
    fetch_thread.start()

def llm_init(llm_type="", api_key="", custom_api="", stream=0):
    global llm, stream_output, CustomLLM, OpenAI

    if CustomLLM == None or OpenAI == None:
        import_deps()

    if stream == "1":
        stream_output = True
    else:
        stream_output = False

    if llm_type == "openai":
        open_api_key = os.getenv('OPENAI_API_KEY') or safe_vim_eval('g:nvim_ai_apikey')
        if open_api_key == "" or open_api_key == None:
            print("OpenAI APIKEY 未定义，请定义 OPENAI_API_KEY")
            return
        llm = OpenAI(openai_api_key=open_api_key, temperature=0.9)
    else:
        llm = CustomLLM(llm_type=llm_type,
                        api_key=api_key,
                        custom_api=custom_api,
                        stream_output=stream_output)

def llm_request(prompt, llm):
    prompts = "\n".join(prompt)
    result = llm(prompts)
    if result == "{error}" or result == "{timeout}":
        return result

    results = result.split("\n")
    return_str = str(results).replace("\\'", "''")
    return return_str

# 调用入口
def just_do_it(prompt):
    global llm
    result = llm_request(prompt, llm)
    if result == "{timeout}" or result == "{error}":
        return '""'
    else:
        return result

if __name__ == '__main__':
    llm_init(llm_type="api2d", api_key="fk209055-QCO1ChYkdCcPi1OnTWss7UlAjifaQ5RU", stream="1")
    print(just_do_it("基于 python 写一段代码，实现一个 helloworld"))

# vim:ts=4:sw=4:sts=4
