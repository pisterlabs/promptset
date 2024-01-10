from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0)


def call_openai_tagging(chain, _sentence):
    from langchain.callbacks import get_openai_callback
    _re = ""
    _tokens = 0
    _cost = 0
    _log = ""
    with get_openai_callback() as cb:
        _re = chain.run(_sentence)
        _tokens = cb.total_tokens
        _cost = cb.total_cost
        _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
        _log += f"Cost: ${cb.total_cost}\n\n"
    print(_sentence, _re)
    print(_log)
    return [_re, _tokens, _cost, _log]

def competitor_openai_tagging(txt_lines):
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0)
    #####
    schema = {
        "properties": {
            "competitor": {
                "type": "string",
                "description": "The specific names or brands of competitors (well-known organizations or companies). If no information is available for 'competitors', output an empty string.",
            },
        },
        "required": ["competitor"],
    }
    chain = create_tagging_chain(schema, llm)
    ##### split notes to sentences
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
    # print(len(_sentences))
    ##### 
    _log = ""
    _total_cost = 0
    _competitor = []
    ##### call OpenAI API with _content and _example
    _log += "-" * 40 + "\n"
    for i in range(0, len(_sentences)):
        [i_re, i_tokens, i_cost, i_log] = call_openai_tagging(chain, _sentences[i])
        _log += i_log
        _total_cost += i_cost
        _competitor.append(i_re)
    _total_cost_str = format(_total_cost, ".5f")
    # print(len(_competitor))
    # print(_competitor)
    ##### parse response, generate _log and _competitor_str
    _log += "-" * 40 + "\n"
    _log += str(_competitor) + "\n"
    _log += "-" * 40 + "\n"
    _log += f"\nTotal Cost: ${_total_cost_str}\n"
    _competitor_str = ""
    if len(_sentences) == len(_competitor):
        _competitor_str = str(_competitor)
    else:
        _log += "Error: len(sentences) != len(7P)" + "\n"
    return [_log, _competitor_str, _total_cost_str, _sentences]

def competitor_llm_tagging(_txt):
    import re
    _log = ""
    _competitor_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _competitor_str, _total_cost_str, _sentences] = competitor_openai_tagging(txt_lines)
    # print(_log)
    # print(_competitor_str)
    import ast
    _competitor = ast.literal_eval(_competitor_str)
    # print(type(_competitor), _competitor)
    return [_competitor, _total_cost_str]


# _txt = "Ved ikke om de har noget organisk affald... på deres hovedkontor har de et køkken, men det er en ekstern operatør der driver det... det er Michael Kjær fra driften, et fælles køkken med andre virksomheder.. Ring til ham om det. NCC bestemmer desuden selv om de skal have vores projekt med i loopet på dgnb point i byggeriet... i deres koncept udvikling...; De er ved at definere det og vi kan vende retur til Martin i Januar, hvor han ved hvem vi skal have møde med om det."

_txt = "They are interested in talking to us, because they think we have a good price, although they use Frederik Machinery already"

[_re, _cost] = competitor_llm_tagging(_txt)
print(type(_re))
for i in _re:
    print(i)
print(type(_cost), _cost)


