import pprint as pp


key = "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"
N_batch = 3


def call_openai(chain, _content):
    from langchain.callbacks import get_openai_callback
    _re = ""
    _tokens = 0
    _cost = 0
    _log = ""
    with get_openai_callback() as cb:
        _re = chain.run(_content=_content)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
        _log += f"Cost: ${cb.total_cost}\n\n"
    # print(_re)
    return [_re, _tokens, _cost, _log]

def split_note_to_sentences(txt_lines):
    ##### split note to sentences
    _sentences = []
    for i in txt_lines:
        i_li = i.strip()
        if i_li:
            for j in i_li.split(". "):
                jj = ""
                if j[-1] == '.':
                    jj = j
                else:
                    jj = j+"."
                _sentences.append(jj)
    return _sentences



def P7_openai(key, txt_lines, N_batch):
    import os
    import re
    from langchain import OpenAI, PromptTemplate, LLMChain
    _log = ""
    _7P_str = ""
    _total_cost = 0
    ##### set OpenAI API Key and prompt
    os.environ["OPENAI_API_KEY"] = key
    llm = OpenAI(temperature=0)
    template = """
Ignore previous instructions. As a marketing strategy analyst, your task is to identify and extract the 7Ps from each customer comment using nouns, according to the 7Ps Marketing Mix.

The customer comments that require marketing strategy analysis are as follows:
{_content}

For each comment, identify and extract relative nouns of 7Ps from and only from the comment. Output the analysis result of each comment in JSON format in one line, with the 7Ps as the main key and the corresponding nouns as the values. The order of the main key is: Product, Price, Place, Promotion, People, Process, Physical evidence.

Please output the analysis results in English lowercase:
"""
    prompt = PromptTemplate(
        input_variables=["_content"],
        template=template,
    )
    ##### LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    ##### split note to sentences
    _sentences = split_note_to_sentences(txt_lines)
    ##### call OpenAI API with _content
    all_re = ""
    for i in range(0, len(_sentences)):
        if i % N_batch == 0:
            batch = _sentences[i:i+N_batch]
            # print(batch)
            _content = ""
            n = int(i / N_batch)
            for j in range(0, len(batch)):
                _content = _content + f"{n*N_batch +j +1}) {batch[j]}\n"
            _log += _content
            # print(prompt.format(_content=_content))
            [b_re, b_tokens, b_cost, b_log] = call_openai(chain, _content)
            _log += b_log
            _total_cost += b_cost
            all_re += b_re + "\n"
            # print(b_re)
    _total_cost_str = format(_total_cost, ".5f")
    _7P_str = all_re.strip()
    return [_log, _7P_str, _total_cost_str, _sentences]


def parse_7P_str(_str, _sentences):
    import re
    import json
    _re = []
    _li = _str.split("\n")
    for i in _li:
        if i:
            # print(i)
            _1 = i.split(" {")
            _i = "{" + _1[1]
            # print(f"_i: {_i}")
            _re.append(_i)
    for i in range(len(_sentences)):
        # print(f"\n{i}, {_sentences[i]}")
        _i_json = json.loads(_re[i])
        # print(type(_i_json), _i_json)
        for j in _i_json:
            if _i_json[j].lower() not in _sentences[i].lower():
                # print(">>>", _i_json[j])
                _i_json[j] = ''
            if _i_json[j].lower() == 'none':
                _i_json[j] = ''
        _re[i] = json.dumps(_i_json, ensure_ascii=False)
        # print(type(_re[i]), _re[i], "\n")
    _re_str = '[' + ', '.join(_re) + ']'
    return _re_str


def P7_llm(_txt):
    global key
    _log = ""
    _7P_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _7P_str, _total_cost_str, _sentences] = P7_openai(key, txt_lines, N_batch)
    # print(_log)
    # print(_7P_str)
    import ast
    _7P_str = parse_7P_str(_7P_str, _sentences)
    # print(_7P_str)
    _7P = ast.literal_eval(_7P_str)
    # print(type(_7P), _7P)
    return [_7P, _total_cost_str]



if __name__ == "__main__":

    _txt = "Ved ikke om de har noget organisk affald... på deres hovedkontor har de et køkken, men det er en ekstern operatør der driver det... det er Michael Kjær fra driften, et fælles køkken med andre virksomheder.. Ring til ham om det. NCC bestemmer desuden selv om de skal have vores projekt med i loopet på dgnb point i byggeriet... i deres koncept udvikling...; De er ved at definere det og vi kan vende retur til Martin i Januar, hvor han ved hvem vi skal have møde med om det."
    [_re, _cost] = P7_llm(_txt)
    print("\n>>>", type(_re), _re)
    print(type(_cost), _cost)
