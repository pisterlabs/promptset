from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0)


# schema = {
#     "properties": {
#         "sentiment": {"type": "string"},
#         "aggressiveness": {"type": "integer"},
#         "reason": {"type": "string"},
#     }
# }
# chain = create_tagging_chain(schema, llm)
# print(chain.run("Ok that I call and find out how it goes."))

# from enum import Enum
# from pydantic import BaseModel, Field
# class Tags(BaseModel):
#     sentiment: str = Field(
#         ...,
#         enum=["positive", "neutral", "negative"],
#         description = "sentiment classification"
#     )
#     aggressiveness: int = Field(
#         ...,
#         enum=[1, 2, 3, 4, 5],
#         description="describes how aggressive the statement is, the higher the number the more aggressive",
#     )
#     reason: str = Field(
#         ...,
#         description="short classification reason"
#     )
# chain = create_tagging_chain_pydantic(Tags, llm)
# print(chain.run("med opstart af SOSU Nord mødte jeg to damer, som var på inspirations tur det nye køkken på UCN."))

# schema = {
#     "properties": {
#         "sentiment": {
#             "type": "string",
#             "enum": ["positive", "neutral", "negative"],
#             "description": "sentiment classification",
#         },
#         "aggressiveness": {
#             "type": "integer",
#             "enum": [1, 2, 3, 4, 5],
#             "description": "describes how aggressive the statement is, the higher the number the more aggressive",
#         },
#         "reason": {
#             "type": "string",
#             "description": "short classification reason",
#         },
#     },
#     "required": ["reason", "sentiment", "aggressiveness"],
# }
# from langchain.callbacks import get_openai_callback
# _re = ""
# _tokens = 0
# _cost = 0
# _log = ""
# chain = create_tagging_chain(schema, llm)
# with get_openai_callback() as cb:
#     _re = chain.run("Estoy muy enojado con vos! Te voy a dar tu merecido!på interesse, tidligere udtrykt, at de ikke var interesseret grundet andre interessepunkter.")
#     _tokens = cb.total_tokens
#     _cost = cb.total_cost
#     _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
#     _log += f"Cost: ${cb.total_cost}\n\n"
#     print(_re)
#     print(_log)


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
    _re = f"{_re['sentiment']}({_re['scale']}: {_re['why']})"
    print(_sentence, _re)
    print(_log)
    return [_re, _tokens, _cost, _log]

def sentiment_openai_tagging(txt_lines):
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0)
    #####
    schema = {
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
                "description": """
The sentiment classification used to judge whether customers have a positive attitude towards the products we are trying to sell to them. In addition to the general sentiment analysis principles, the following rules must be followed: 1) If the main content of the comment involves numbers (phone numbers, dates, addresses, web addresses, etc.), it is neutral; 2) If the main content of the comment is dominated by interjections, modal particles, nouns or adjectives with no obvious emotional meaning, it is neutral; 3) If the customer is willing to communicate or agrees to our call back, it's positive; 4) If the customer wants to talk to technicians, it's positive; 5) If the customer is likely to purchase in the future, it's positive; 6) If the customer is talking about invoiced, delivered, and billing, it's positive; 7) If the customer is discussing personal schedule and activities, it is positive.

Below are some negative and positive examples of sentiment analysis for customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the sentiment classification of the comments:
"Haven't chosen a system for Gødstrup, if it were to start before ex Herning, it would have to be the same system as it will be in Gødstrup.", negative
"på interesse, tidligere udtrykt, at de ikke var interesseret grundet andre interessepunkter.", negative
"We are in good shape if it turns out to be a grind.", positive
"Note that we are happy to arrange a demonstration.", positive
"Ok that I call and find out how it goes.", positive
"Opfølgning på målinger i de andre butikker.", positive
"på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen.", positive
"Den i Bones kan tages retur efter 2 måneder, den anden er købt.", positive
indkøbes/faktureres og leveres ca.", positive
"Møde med Richard og to teknikere.", positive
"Meget positivt møde.", positive
"Der skal fremsendes 3 tilbud: 1) Kværn i opvask, rørføring og tank i skakt og sugeledning ud til ydrevæg.", positive
"2) Kværn i grøntrum, rørføring ved trappe og isoleret tank i hækken.", positive
"3) Kværn i grøntrum, rørføring ved trappe og nedgravet tank under hækken.", positive
"Fundet uge 46 Eva Ejlsskov, måske lederne 11.08.14 JJ De er interesseret i lejeløsning.?.", positive
"Hej Kirsten Vi talte sammen først på året, og jeg kan fortælle, at vi nu har omkring 40 anlæg stående – disse er fordelt i hele landet, det nærmeste anlæg er hos Legoland i Billund, hvor det evt.", positive
"Har I noget nyt med hensyn til ombygning i køkkenet ? Venlig Hilsen Jens Jeberg Biotrans Nordic Svendborgvej 243 DK – 5260 Odense SMob.", positive
"15/8, hvor vi er i Århus alligevel.", positive
"opfølgning på tilbud om leje og køb - sendt mail d.", positive
"med opstart af SOSU Nord mødte jeg to damer, som var på inspirations tur det nye køkken på UCN.", positive
"De var meget interesserede og fik materiale med hjem.", positive
"Men hvis den ikke blev godkendt i år, så bliver den måske næste år.", positive
"Den ligger fortsat i ansøgningerne, en ansøgning som de har levet, og stadig er meget positiv for.", positive
""",
            },
            "why": {
                "type": "string",
                "description": "The short reason of sentiment classification in English lowercase.",
            },
            "scale": {
                "type": "integer",
                "enum": [1, 2, 3, 4, 5],
                "description": "Describes how aggressive the statement is, the higher the number the more aggressive.",
            },
        },
        "required": ["sentiment", "why", "scale"],
        # "required": ["sentiment", "reason"],
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
    _sentiments = []
    ##### call OpenAI API with _content and _example
    _log += "-" * 40 + "\n"
    for i in range(0, len(_sentences)):
        [i_re, i_tokens, i_cost, i_log] = call_openai_tagging(chain, _sentences[i])
        _log += i_log
        _total_cost += i_cost
        _sentiments.append(i_re)
    _total_cost_str = format(_total_cost, ".5f")
    # print(len(_sentiments))
    ##### parse response, generate _log, _sentences_str and _sentiments_str
    _log += "-" * 40 + "\n"
    _log += "\n".join(_sentiments) + "\n"
    _log += "-" * 40 + "\n"
    _log += f"\nTotal Cost: ${_total_cost_str}\n"
    _sentences_str = ""
    _sentiments_str = ""
    if len(_sentences) == len(_sentiments):
        _sentences_str = "\n".join(_sentences)
        _sentiments_str = "\n".join(_sentiments)
    else:
        _log += "Error: len(sentences) != len(sentiments)" + "\n"
    return [_log, _sentences_str, _sentiments_str, _total_cost_str]

def sentiment_llm_tagging(_txt):
    import re
    _log = ""
    _sentences_str = ""
    _sentiments_str = ""
    _total_cost = 0
    txt_lines = _txt.split("\n")
    [_log, _sentences_str, _sentiments_str, _total_cost] = sentiment_openai_tagging(txt_lines)
    # print(_log)
    _out = []
    if _sentences_str != "" and _sentiments_str != "":
        sentences = _sentences_str.split("\n")
        sentiments = _sentiments_str.split("\n")
        if len(sentences) == len(sentiments):
            for i in range(0, len(sentences)):
                # i_re = f"{sentences[i]}|{sentiments[i]}\n"
                _out.append(re.sub('\d+\)\s+', '', sentiments[i]))
            # print(f"return:\n{_out}")
        else:
            print("Error: len(sentences) != len(sentiments)")
    return [_out, str(_total_cost)]


# _txt = """Opfølgning på målinger i de andre butikker. Obs. på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen. Desuden obs på, at det er meningen, det skal opbevares i længere tid.
# .med en basic og en SLIM. Den i Bones kan tages retur efter 2 måneder, den anden er købt.
# Tank leveres asap Vi borer huller og trækker rør ind i bygningen Kværnen indkøbes/faktureres og leveres ca. 1/4-15
# """

_txt = """Haven't chosen a system for Gødstrup, if it were to start before ex Herning, it would have to be the same system as it will be in Gødstrup.
på interesse, tidligere udtrykt, at de ikke var interesseret grundet andre interessepunkter.
We are in good shape if it turns out to be a grind.
Note that we are happy to arrange a demonstration.
Ok that I call and find out how it goes.
Opfølgning på målinger i de andre butikker.
på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen.
Den i Bones kan tages retur efter 2 måneder, den anden er købt.
indkøbes/faktureres og leveres ca.
Møde med Richard og to teknikere.
Meget positivt møde.
Der skal fremsendes 3 tilbud: 1) Kværn i opvask, rørføring og tank i skakt og sugeledning ud til ydrevæg.
2) Kværn i grøntrum, rørføring ved trappe og isoleret tank i hækken.
3) Kværn i grøntrum, rørføring ved trappe og nedgravet tank under hækken.
Fundet uge 46 Eva Ejlsskov, måske lederne 11.08.14 JJ De er interesseret i lejeløsning.?.
Hej Kirsten Vi talte sammen først på året, og jeg kan fortælle, at vi nu har omkring 40 anlæg stående – disse er fordelt i hele landet, det nærmeste anlæg er hos Legoland i Billund, hvor det evt.
Har I noget nyt med hensyn til ombygning i køkkenet ? Venlig Hilsen Jens Jeberg Biotrans Nordic Svendborgvej 243 DK – 5260 Odense SMob.
15/8, hvor vi er i Århus alligevel.
opfølgning på tilbud om leje og køb - sendt mail d.
med opstart af SOSU Nord mødte jeg to damer, som var på inspirations tur det nye køkken på UCN.
De var meget interesserede og fik materiale med hjem.
Men hvis den ikke blev godkendt i år, så bliver den måske næste år.
Den ligger fortsat i ansøgningerne, en ansøgning som de har levet, og stadig er meget positiv for.
Hanne sits in the working group regarding Gødstrup, among other things, waste.
Decision on standby.
Order up from evaluating different systems.
Our material is included.
I also suggest talking to Ole Teglgård.
Obs.
Desuden obs på, at det er meningen, det skal opbevares i længere tid.
.med en basic og en SLIM.
Tank leveres asap Vi borer huller og trækker rør ind i bygningen Kværnen 1/4-15.
Pt.
samler de affaldet i kantinen på 4 sal, kører det i poser ned i affaldsrummet og tømmer dem i blå spande fra M.
Larsen.
Forsøger kontakt mhp.
kan besigtiges.
+45 22 15 25 09 Tel.
+45 70 25 84 00 www.biotrans-nordic.com.
Startet med linked in invitation til direktør.
Ligger lige ved siden af Agrotech, måske de har været involveret.
Satser på møde måske d.
1/11.
Ifm.
Liselotte Kirk 25676514 og Lone Holm 51901534 fra LL Catering.
Opfølgning på tilbud fra sidste år.
Talte med Lars d.
23/10 - og han sagde at den lå i køkkenets ansøgninger og at ikke længere kunne trykke den frem.
Det er bare deres systems gang.
"""

[_re, _cost] = sentiment_llm_tagging(_txt)
print(type(_re))
for i in _re:
    print(i)
print(type(_cost), _cost)


