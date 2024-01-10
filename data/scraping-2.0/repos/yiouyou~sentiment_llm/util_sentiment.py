##### v0
# import pprint as pp
# key = "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"
# N_batch = 5

# def call_openai(chain, _content, _example):
#     from langchain.callbacks import get_openai_callback
#     _re = ""
#     _tokens = 0
#     _cost = 0
#     _log = ""
#     with get_openai_callback() as cb:
#         _re = chain.run(_content=_content, _example=_example)
#         _tokens = cb.total_tokens
#         _cost = cb.total_cost
#         _log += f"\nTokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens})\n"
#         _cost_str = format(cb.total_cost, ".5f")
#         _log += f"Cost: ${_cost_str}\n\n"
#     # print(_re)
#     return [_re, _tokens, _cost, _log]

# def sentiment_openai(key, txt_lines, N_batch):
#     import os
#     import re
#     from langchain import OpenAI, PromptTemplate, LLMChain
#     _log = ""
#     _sentences_str = ""
#     _sentiments_str = ""
#     _total_cost = 0
#     ##### set OpenAI API Key and prompt
#     os.environ["OPENAI_API_KEY"] = key
#     llm = OpenAI(temperature=0)
#     template = """
# Ignore previous instructions. You are a sentiment analyst of customer comments. You assist the company in further operations by dividing customer comments into three categories: positive, negative and neutral. The main purpose is to judge whether customers have a positive attitude towards the products we are trying to sell to them. When analyzing comments, in addition to the general sentiment analysis principles, the following rules must be followed:
# 1) If the customer is likely to agree to our call back, it is considered positive
# 2) If the customer is willing to communicate further or is likely to purchase in the future, it is considered positive
# 3) If the main content of the comment involves numbers, phone numbers, dates, addresses or web addresses, it is considered neutral
# 4) If the main content of the comment is dominated by interjections, modal particles, nouns or adjectives with no obvious emotional meaning, it is considered neutral

# Below are some examples of sentiment analysis for some customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the sentiment classification of the comments:
# {_example}

# The customer comment texts that require sentiment analysis are as follows:
# {_content}

# For each comment, there is no need to output the comment itself, just output the comment index, sentiment classification and short classification reason in the format of "index) classification(reason)", and output the analysis results in English lowercase:
# """
#     prompt = PromptTemplate(
#         input_variables=["_content", "_example"],
#         template=template,
#     )
#     ##### 随机取10个example
#     import random
#     with open("examples_sentiment.txt", "r", encoding="utf-8") as ef:
#         _example = "".join(random.sample(ef.readlines(), 30))
#     ##### LLMChain
#     chain = LLMChain(llm=llm, prompt=prompt)
#     ##### split comment to sentences
#     _sentences = []
#     for i in txt_lines:
#         i_li = i.strip()
#         if i_li:
#             for j in i_li.split(". "):
#                 jj = ""
#                 if j[-1] == '.':
#                     jj = j
#                 else:
#                     jj = j+"."
#                 _sentences.append(jj)
#     ##### call OpenAI API with _content and _example
#     _log += "-" * 40 + "\n"
#     all_re = ""
#     for i in range(0, len(_sentences)):
#         if i % N_batch == 0:
#             batch = _sentences[i:i+N_batch]
#             # print(batch)
#             _content = ""
#             n = int(i / N_batch)
#             for j in range(0, len(batch)):
#                 _content = _content + f"{n*N_batch +j +1}) {batch[j]}\n"
#             _log += _content
#             # print(prompt.format(_content=_content, _example=_example))
#             [b_re, b_tokens, b_cost, b_log] = call_openai(chain, _content, _example)
#             _log += b_log
#             _total_cost += b_cost
#             all_re += b_re + "\n"
#     ##### parse response, generate _log, _sentences_str and _sentiments_str
#     sentences = []
#     sentiments = []
#     all_re = re.sub(r" *\(", " (", all_re.lower())
#     all_re = re.sub(r"\n+", "\n", all_re)
#     _sentiments = all_re.split("\n")
#     for i in _sentiments:
#         if i != "":
#             sentiments.append(i)
#     _log += "-" * 40 + "\n"
#     _log += "\n".join(sentiments) + "\n"
#     _log += "-" * 40 + "\n"
#     _total_cost_str = format(_total_cost, ".5f")
#     _log += f"\nTotal Cost: ${_total_cost_str}\n"
#     if len(_sentences) == len(sentiments):
#         for i in range(0, len(_sentences)):
#             sentences.append(f"{i+1}) \"{_sentences[i]}\"")
#         _sentences_str = "\n".join(sentences)
#         _sentiments_str = "\n".join(sentiments)
#     else:
#         _log += "Error: len(sentences) != len(sentiments)" + "\n"
#     return [_log, _sentences_str, _sentiments_str, _total_cost_str]

# def sentiment_llm(_txt):
#     global key
#     import re
#     _log = ""
#     _sentences_str = ""
#     _sentiments_str = ""
#     _total_cost = 0
#     txt_lines = _txt.split("\n")
#     [_log, _sentences_str, _sentiments_str, _total_cost] = sentiment_openai(key, txt_lines, N_batch)
#     # print(_log)
#     _out = []
#     if _sentences_str != "" and _sentiments_str != "":
#         sentences = _sentences_str.split("\n")
#         sentiments = _sentiments_str.split("\n")
#         if len(sentences) == len(sentiments):
#             for i in range(0, len(sentences)):
#                 # i_re = f"{sentences[i]}|{sentiments[i]}\n"
#                 _out.append(re.sub('\d+\)\s+', '', sentiments[i]))
#             # print(f"return:\n{_out}")
#         else:
#             print("Error: len(sentences) != len(sentiments)")
#     return [_out, str(_total_cost)]



##### v1
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
        _cost_str = format(cb.total_cost, ".5f")
        _log += f"Cost: ${_cost_str}\n\n"
    # _re = f"{_re['sentiment']}({_re['scale']}: {_re['why']})"
    _re = f"{_re['sentiment']}({_re['why']})"
    print(_sentence, _re)
    print(_log)
    return [_re, _tokens, _cost, _log]

def sentiment_openai_tagging(txt_lines):
    from langchain.chains import create_tagging_chain
    from dotenv import load_dotenv
    load_dotenv()
    from langchain.chat_models import ChatOpenAI
    import os
    llm = ChatOpenAI(temperature=0, model_name=os.getenv('OPENAI_MODEL'))
    #####
    schema = {
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
                "description": """
The sentiment classification used to judge whether customers have a positive attitude towards the products we are trying to sell to them. In addition to the general sentiment analysis principles, the following rules must be followed: 
1) If the main content of the comment involves numbers (phone numbers, dates, addresses, web addresses, etc.), it is neutral;
2) If the main content of the comment is dominated by interjections, modal particles, nouns or adjectives with no obvious emotional meaning, it is neutral;
3) If the customer doesn't need our products, it is negative;
4) If the customer indicates having an existing channel or their own product, it is negative;
5) If the customer says negatively that they do not have certain product, it is negative;
6) If the customer says the store is closed, it is negative;
7) If the customer shows willingness to communicate, agrees to a callback, seeks to talk to technicians, or seeks more information, it's positive;
8) If the customer requests or talks about a meeting or asks us to do something for them, it's positive;
9) If the customer is likely to purchase in the future, it's positive;
10) If the customer is talking about invoiced, delivered, and billing, it's positive;
11) If the customer is discussing personal schedule and activities, it is positive;
12) If the customer discusses an ongoing or future project or or expresses interest in joining larger projects, it is positive;
13) If the customer is willing to cooperate with our requirements, such as confirming the location or renovating the space of our product, etc., it is positive.

Below are some negative and positive examples of sentiment analysis for customer comments in csv format, where the customer's comments are enclosed in double quotes, and after the comma is the sentiment classification of the comments:
"De er godt presset lige nu og skal bare sælge de varer de har", negative
"De har valgt, at de flytter de eksisterende møbler med i den nye lokaler til at starte med", negative
"Haven't chosen a system for Gødstrup, if it were to start before ex Herning, it would have to be the same system as it will be in Gødstrup.", negative
"Henrik, vi er medlem af en indkøbsforening og de får deres egen producerede varer den vej igennem", negative
"Intet madaffald siger Charlotte", negative
"Overhovedet, har ingen madaffald, da de ingen kantineordning har", negative
"Regnskabet i firmaet som dækker over 1½ år afslører at der er 0 ansatte og der har været en OMS på i alt 1000kr. Så der sker tilsyneladende ikke meget i det firma. Det er stadig Klaudia og Giulia der arbejder på at få firmaet i luften.", negative
"Vores tøj laves på egne fabrikker i Tyrkiet, han regner ikke med at vi kan være med på kvaliteten, desuden er de forpligtet til at bruge deres fabrikker", negative
"han ringede tilbagede lavede selv deres tøj", negative
"har hun ikke brug da det er en second hand, hun har", negative
"havde lukket sin tøj butik", negative
"kontakt kontoret ring til kæde chef er ved at skære ned på deres leverandører og vil være gode ved dem de har wunderwear samba se om jeg kan finde noget ellers ring igen så har hun et nummer", negative
"på interesse, tidligere udtrykt, at de ikke var interesseret grundet andre interessepunkter.", negative
"susanne ejer butikken det gad hun ikke", negative
"15/8, hvor vi er i Århus alligevel.", positive
"Der skal fremsendes 3 tilbud: 1) Kværn i opvask, rørføring og tank i skakt og sugeledning ud til ydrevæg.", positive
"2) Kværn i grøntrum, rørføring ved trappe og isoleret tank i hækken.", positive
"3) Kværn i grøntrum, rørføring ved trappe og nedgravet tank under hækken.", positive
"Burde komme midt uge 38. 18.9: Podier er kommet.", positive
"De skal finde ud af om de vil have det store eller det lille anlæg. Han regner med at de køber solceller af os. inden for 1 måneds tid kommer der en beslutning.", positive
"De var meget interesserede og fik materiale med hjem.", positive
"Den i Bones kan tages retur efter 2 måneder, den anden er købt.", positive
"Den ligger fortsat i ansøgningerne, en ansøgning som de har levet, og stadig er meget positiv for.", positive
"Der er 10! Placering af kværn i køkken virker mulig og rør trækkes over det forsænkdede loft ud til plads, hvor daka spande står idag", positive
"Det er ikke fordi, at de ikke vil have nye møbler i forbindelse med flytningen, men de har ikke haft tid til at gå træffe den endelige beslutning omkring de nye møbler", positive
"Fint møde: Maria sender 3D filer og info om 3D scanning Evt. teams møde uge 39", positive
"Fundet uge 46 Eva Ejlsskov, måske lederne 11.08.14 JJ De er interesseret i lejeløsning.?.", positive
"Har I noget nyt med hensyn til ombygning i køkkenet ? Venlig Hilsen Jens Jeberg Biotrans Nordic Svendborgvej 243 DK – 5260 Odense SMob.", positive
"Hej Bo, Jeg har nu spurgt producenten om et tilbud på shaft sleeve pos. 6605 uden coating/Chrome oxid.Jeg holder dig opdateret.", positive
"Hej Dag, Tak for et godt møde den 6. september.På mødet talte vi om genindvinding af jernsulfat fra jeres spildevand, på en mere økonomisk måde, ved hjælp af vores filterløsning frem for inddampning eller nedkøling.I den forbindelse foreslog vi at få en spildevandsprøve tilsendt således at vi kan teste det i vores filterløsning.Desværre har er vi ikke selv mulighed for at komme forbi jer og hente en spildevandsprøve.Lad os venligst vide om i har mulighed for sende en prøve til os.På forhånd tak og god weekend.", positive
"Hej Kirsten Vi talte sammen først på året, og jeg kan fortælle, at vi nu har omkring 40 anlæg stående – disse er fordelt i hele landet, det nærmeste anlæg er hos Legoland i Billund, hvor det evt.", positive
"Hej NilsJeg er blevet spurgt om der er belægning på shaft sleeve pos. 6605Hvis der er det, har vi brug for certifikat på at det er godkendt til fødevare produktion.Er det noget som du kan svare på?", positive
"Hun vil lige se på Pitaya hjemmesidem, der er alt tøj fra os", positive
"Lars oplyste, at Susanne sidder med beslutningen, og i forhold til det nye OUH oplyste han, at intet var besluttet, men han kunne se at de gør plads til Køkken i de nyeste tegninger, muligvis en identifikation eller blot en dør på klem", positive
"Meget positivt møde.", positive
"Men hvis den ikke blev godkendt i år, så bliver den måske næste år.", positive
"Modtaget svar og ny forespørgsel. "Hej Nils.Jeg vender som sagt tilbage, når jeg får et endeligt svar fra dem med pengene. Vedhæftet det røreværk, hvor jeg gerne vil have et tilbud på en ny aksel.Spørgsmål: Bør der ikke være et lille overtryk på akseltætningen ESD34, da der er 2-3 meter væskesøjle over tætningen? "", positive
"Møde med Richard og to teknikere.", positive
"Mødet er booket indtil videre med Morten.Desuden; Vi kører nogle projekter på Arla, IT standarder og automationsstandarder, navne standarder bliver det første. De afholder 2 timers undervisning i Viby med automationsfolk fra alle mulige steder og de vil gerne invitere vores kontakt til Arla ind for at være med i den undervisning så vi bedre forstår hvad Arla skal bruge.", positive
"Note that we are happy to arrange a demonstration.", positive
"Ok that I call and find out how it goes.", positive
"Opfølgning på Norge, Ny Dobbelthal samt forbrig, 30 nye stationer i dk. samt 4g dongle i Ikast", positive
"Opfølgning på målinger i de andre butikker.", positive
"Projektet er stadig aktivt omkring kværn, men det er blevet en del af et større projekt og trækker derfor ud", positive
"Projektet er udskudt til næste år, men stadig i gang", positive
"Rikke@libenodic.dk vil gerne have noget konkret på hvad vi har lavet og nogle pris eksempler hvilke systuer arbejder vi med. er det nogle ordenlige forhold de arbejder under, (det er vigtigt) er det nogle kontrolleret forhold de arbejder under?", positive
"Skal først bruge møbler til juli, har aftalt at jeg sender min kontakt oplysninger", positive
"Teamsmøde sammen med Jesper vedr. løsninger og batteri vi opdatere løsningen.", positive
"Vil gerne have et møde 22/8/23 kl", positive
"We are in good shape if it turns out to be a grind.", positive
"aftalt møde i uge 28", positive
"det har ikke interesse lige numåske til nogle sommer kjoler", positive
"en start pris, på en t-shirt spændene er tilbage fra møde den 3 kl 15 send mail", positive
"er interresseret mandag den 18 kl 10 er fedt med kort leveringstid og lavt antal bestillinger, hvilket også ville være godt med børnetøj er lidt skeptisk, men vil gerne have et møde og høre mere om det", positive
"indkøbes/faktureres og leveres ca.", positive
"kan vi sælge en t-shirt til 100 kr med tryk på vil se noget først, tag nogle vareprøver med han er i butikken hverdag så vi skal bare stikke hovedet ind vi skal ikke køre for det, så det skal være hvis vi skal der ned alligevel", positive
"med opstart af SOSU Nord mødte jeg to damer, som var på inspirations tur det nye køkken på UCN.", positive
"opfølgning på tilbud om leje og køb - sendt mail d.", positive
"projektleder fra ISS skal køre hele husets affaldsprojekt samtidig, så han forventer et efterårs projekt", positive
"på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen.", positive
"ring onsdag, torsdag eller fredag skal have fat Michael han står for indkøb og er medejerehan er tilbage mandag om 8 dage okse rød skjorte ca pris på den skorte", positive
"send info priser kvaliteter var interesseret vil gerne have noget info og piser", positive
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
        i_li = i_li.replace("&nbsp; ", "").replace("&nbsp;", "").replace("<a href=\"mailto:", "")
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



if __name__ == "__main__":

#     _txt = """Opfølgning på målinger i de andre butikker. Obs. på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen. Desuden obs på, at det er meningen, det skal opbevares i længere tid.
# .med en basic og en SLIM. Den i Bones kan tages retur efter 2 måneder, den anden er købt.
# Tank leveres asap Vi borer huller og trækker rør ind i bygningen Kværnen indkøbes/faktureres og leveres ca. 1/4-15
# """
#     [_re, _cost] = sentiment_llm(_txt)
#     print(type(_re), _re)
#     print(type(_cost), _cost)

#     _txt = """Haven't chosen a system for Gødstrup, if it were to start before ex Herning, it would have to be the same system as it will be in Gødstrup.
# på interesse, tidligere udtrykt, at de ikke var interesseret grundet andre interessepunkter.
# We are in good shape if it turns out to be a grind.
# Note that we are happy to arrange a demonstration.
# Ok that I call and find out how it goes.
# Opfølgning på målinger i de andre butikker.
# på at der altid er lidt mere og obs på at det er efterårsferie, hvis det har noget at sige for omsætningen.
# Den i Bones kan tages retur efter 2 måneder, den anden er købt.
# indkøbes/faktureres og leveres ca.
# Møde med Richard og to teknikere.
# Meget positivt møde.
# Der skal fremsendes 3 tilbud: 1) Kværn i opvask, rørføring og tank i skakt og sugeledning ud til ydrevæg.
# 2) Kværn i grøntrum, rørføring ved trappe og isoleret tank i hækken.
# 3) Kværn i grøntrum, rørføring ved trappe og nedgravet tank under hækken.
# Fundet uge 46 Eva Ejlsskov, måske lederne 11.08.14 JJ De er interesseret i lejeløsning.?.
# Hej Kirsten Vi talte sammen først på året, og jeg kan fortælle, at vi nu har omkring 40 anlæg stående – disse er fordelt i hele landet, det nærmeste anlæg er hos Legoland i Billund, hvor det evt.
# Har I noget nyt med hensyn til ombygning i køkkenet ? Venlig Hilsen Jens Jeberg Biotrans Nordic Svendborgvej 243 DK – 5260 Odense SMob.
# 15/8, hvor vi er i Århus alligevel.
# opfølgning på tilbud om leje og køb - sendt mail d.
# med opstart af SOSU Nord mødte jeg to damer, som var på inspirations tur det nye køkken på UCN.
# De var meget interesserede og fik materiale med hjem.
# Men hvis den ikke blev godkendt i år, så bliver den måske næste år.
# Den ligger fortsat i ansøgningerne, en ansøgning som de har levet, og stadig er meget positiv for.
# Hanne sits in the working group regarding Gødstrup, among other things, waste.
# Decision on standby.
# Order up from evaluating different systems.
# Our material is included.
# I also suggest talking to Ole Teglgård.
# Obs.
# Desuden obs på, at det er meningen, det skal opbevares i længere tid.
# .med en basic og en SLIM.
# Tank leveres asap Vi borer huller og trækker rør ind i bygningen Kværnen 1/4-15.
# Pt.
# samler de affaldet i kantinen på 4 sal, kører det i poser ned i affaldsrummet og tømmer dem i blå spande fra M.
# Larsen.
# Forsøger kontakt mhp.
# kan besigtiges.
# +45 22 15 25 09 Tel.
# +45 70 25 84 00 www.biotrans-nordic.com.
# Startet med linked in invitation til direktør.
# Ligger lige ved siden af Agrotech, måske de har været involveret.
# Satser på møde måske d.
# 1/11.
# Ifm.
# Liselotte Kirk 25676514 og Lone Holm 51901534 fra LL Catering.
# Opfølgning på tilbud fra sidste år.
# Talte med Lars d.
# 23/10 - og han sagde at den lå i køkkenets ansøgninger og at ikke længere kunne trykke den frem.
# Det er bare deres systems gang.
# """
#     [_re, _cost] = sentiment_llm_tagging(_txt)
#     print(type(_re))
#     for i in _re:
#         print(i)
#     print(type(_cost), _cost)

    _txt_neg = """Vores tøj laves på egne fabrikker i Tyrkiet, han regner ikke med at vi kan være med på kvaliteten, desuden er de forpligtet til at bruge deres fabrikker.
havde lukket sin tøj butik.
han ringede tilbagede lavede selv deres tøj.
Overhovedet, har ingen madaffald, da de ingen kantineordning har.
Intet madaffald siger Charlotte.
De er godt presset lige nu og skal bare sælge de varer de har.
Henrik, vi er medlem af en indkøbsforening og de får deres egen producerede varer den vej igennem.
De har valgt, at de flytter de eksisterende møbler med i den nye lokaler til at starte med.
"""
    [_re, _cost] = sentiment_llm_tagging(_txt_neg)
    print('neg:', type(_re))
    for i in _re:
        print(i)
    print(type(_cost), _cost)
    _txt_pos = """Projektet er udskudt til næste år, men stadig i gang.
projektleder fra ISS skal køre hele husets affaldsprojekt samtidig, så han forventer et efterårs projekt.
Projektet er stadig aktivt omkring kværn, men det er blevet en del af et større projekt og trækker derfor ud.
Der er 10! Placering af kværn i køkken virker mulig og rør trækkes over det forsænkdede loft ud til plads, hvor daka spande står idag.
Lars oplyste, at Susanne sidder med beslutningen, og i forhold til det nye OUH oplyste han, at intet var besluttet, men han kunne se at de gør plads til Køkken i de nyeste tegninger, muligvis en identifikation eller blot en dør på klem.
Hun vil lige se på Pitaya hjemmesidem, der er alt tøj fra os.
aftalt møde i uge 28.
Vil gerne have et møde 22/8/23 kl.
Skal først bruge møbler til juli, har aftalt at jeg sender min kontakt oplysninger.
Det er ikke fordi, at de ikke vil have nye møbler i forbindelse med flytningen, men de har ikke haft tid til at gå træffe den endelige beslutning omkring de nye møbler.
kan vi sælge en t-shirt til 100 kr med tryk på vil se noget først, tag nogle vareprøver med han er i butikken hverdag så vi skal bare stikke hovedet ind vi skal ikke køre for det, så det skal være hvis vi skal der ned alligevel.
en start pris, på en t-shirt spændene er tilbage fra møde den 3 kl 15 send mail.
"""
    [_re, _cost] = sentiment_llm_tagging(_txt_pos)
    print('pos:', type(_re))
    for i in _re:
        print(i)
    print(type(_cost), _cost)

