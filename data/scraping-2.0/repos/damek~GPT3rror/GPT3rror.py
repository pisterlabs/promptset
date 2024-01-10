### Adapted from https://stackoverflow.com/a/40135960

import openai

from IPython.core.ultratb import AutoFormattedTB

# initialize the formatter for making the tracebacks into strings
itb = AutoFormattedTB(mode = 'Plain', tb_offset = 1)


# this function will be called on exceptions in any cell
def custom_exc(shell, etype, evalue, tb, tb_offset=None,
               openai_api_key="",
               openai_model="text-curie-001",
               max_tokens=256,
               pass_cell_to_GPT3=False,
               temperature=.7):
    openai.api_key = openai_api_key

    # still show the error within the notebook, don't just swallow it
    shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    # grab the traceback and make it into a list of strings
    stb = itb.structured_traceback(etype, evalue, tb)
    sstb = itb.stb2text(stb)
    # get the contents of the cell that raised the error
    cell = shell.history_manager.input_hist_parsed[shell.execution_count]

    params = {"model": openai_model,
              "prompt": "",
              "temperature": temperature, "max_tokens": max_tokens, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0}
    prompt = "I received the following error after trying to compile my python code." \
             + sstb + \
             "\n===\n Explain the error and how to fix it. " + \
             "Also write up to 2 suggested corrections to the python code. " + \
             "Use the format 'Suggested Corrections: '"
    if pass_cell_to_GPT3 == True:
        params["prompt"] = "Here is my code: " + cell + "\n===\n" + prompt
    else:
        params["prompt"] = prompt
    response = openai.Completion.create(**params)
    print (sstb)
    print(response["choices"][0]["text"])


def raise_error(openai_api_key="",
                openai_model="text-curie-001",
                max_tokens=256,
                pass_cell_to_GPT3=False,
                temperature=0.7):
    # write a lambda that wraps custom_exc with the arguments you want
    # to pass to it
    custom_exc_wrapped = lambda shell, etype, evalue, tb, tb_offset: \
        custom_exc(shell, etype, evalue, tb, tb_offset, openai_api_key=openai_api_key,
                   openai_model=openai_model, max_tokens=max_tokens,pass_cell_to_GPT3=pass_cell_to_GPT3,
                   temperature=temperature)
    get_ipython().set_custom_exc((Exception,), custom_exc_wrapped)
