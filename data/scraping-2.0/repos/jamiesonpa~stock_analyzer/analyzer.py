
import requests
import os
import chronology
import streamlit as st
import openai

# os.system("python3.7 -m pip install openai")
# os.system("curl -u :"+st.secrets["API_KEY"]+" https://api.openai.com/v1/engines/")

def get_website_name(name):
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    stock_name = name
    response = openai.Completion.create(
    engine="davinci",
    prompt="Q: What is the website URL for Arcturus Therapeutics?\nA: https://arcturusrx.com/\n\nQ: What is the website URL for Ionis Pharmaceuticals?\nA: https://www.ionispharma.com/\n\nQ: What is the website for Allogene Therapeutics?\nA: https://www.allogene.com/\n\nQ: What is the website for Iovance Biotherapeutics?\nA: https://www.iovance.com/\n\nQ: What is the website for Intellia Therapeutics?\nA: https://www.intelliatx.com/\n\nQ: What is the website for Repare Therapeutics?\nA: https://www.reparerx.com/\n\nQ: What is the website for Vertex Pharmaceuticals?\nA: https://www.vrtx.com/\n\nQ: What is the website for Dicerna Pharmaceuticals?\nA: https://www.dicerna.com/\n\nQ: What is the website for Beam Therapeutics?\nA: https://www.beamtx.com/\n\nQ: What is the website for Pfizer?\nA: https://www.pfizer.com/\n\nQ: What is the website for Exact Sciences?\nA: https://www.exactsciences.com/\n\nQ: What is the website for " + stock_name +"?\n",
    temperature=0,
    max_tokens=100,
    top_p=1,
    best_of=10,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n"]
    )
    answer = (str(response).split('"text": "A: ')[1]).split('"')[0]
    retval = str(answer)
    return retval

def get_ticker(name):
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    stock_name = name
    response = openai.Completion.create(
    engine="davinci",
    prompt="Q: What is the ticker for Arcturus Therapeutics?\nA: ARCT\n\nQ: What is the ticker for Ionis Pharmaceuticals?\nA: IONS\n\nQ: What is the ticker for Allogene Therapeutics?\nA: ALLO\n\nQ: What is the ticker for Iovance Biotherapeutics?\nA: IOVA\n\nQ: What is the ticker for Intellia Therapeutics?\nA: NTLA\n\nQ: What is the ticker for Repare Therapeutics?\nA: RPTX\n\nQ: What is the ticker for Vertex Pharmaceuticals?\nA: VRTX\n\nQ: What is the ticker for Dicerna Pharmaceuticals?\nA: DRNA\n\nQ: What is the ticker for Beam Therapeutics?\nA: BEAM\n\nQ: What is the ticker for Pfizer?\nA: PFE\n\nQ: What is the ticker for Exact Sciences?\nA: EXAS\n\nQ: What is the ticker for " + stock_name +"?\n",
    temperature=0,
    max_tokens=100,
    top_p=1,
    best_of=10,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n"]
    )
    answer = (str(response).split('"text": "A: ')[1]).split('"')[0]
    retval = str(answer)
    return retval
    
def get_industry(name):
    stock_name = name
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    response = openai.Completion.create(
    engine="davinci",
    prompt="Q: What type of company is Teladoc Health?\nA: healthcare\n\nQ: What type of company is Arcturus Therapeutics?\nA: pharmaceutical\n\nQ: What type of company is Intellia Therapeutics?\nA: pharmaceutical\n\nQ: What type of company is Benson Hill?\nA: agricultural\n\nQ: What type of company is Personalis?\nA: healthcare\n\nQ: What type of company is Corteva?\nA: agricultural\n\nQ: What type of company is 10X Genomics?\nA: biological tools\n\nQ: What type of company is Pacific Biosciences?\nA: biological tools\n\nQ: What type of company is Illumina?\nA: biological tools\n\nQ: What type of company is Iovance Biotherapeutics?\nA: pharmaceutical\n\nQ: What type of company is Oxford Nanopore?\nA: biological tools'\n\nQ: What type of company is Ginkgo Bioworks?\nA: synthetic biology\n\nQ: What type of company is Amyris?\nA: synthetic biology\n\nQ: What type of company is Aquabounty?\nA: agricultural\n\nQ: What type of company is Ionis Pharmaceuticals?\nA: pharmaceutical\nQ: What type of company is " + stock_name +"?\n",
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,    # ncts = pagetext.split('<td class="responsiveContent">NCT')
    # clean_ncts = []
    # for nct in ncts:
    #     clean_ncts.append(nct[:20])
    # for nct in clean_ncts:
    #     print(nct)


    stop=["\n"]
    )
    answer = (str(response).split('"text": "A: ')[1]).split('"')[0]
    retval = str(answer)
    return retval

def get_diseases(name):
    stock_name = name
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    response = openai.Completion.create(
    engine="davinci",
    prompt="Q: For which diseases is Arcturus Therapeutics trying to develop treatments?\nA: COVID-19, influenza, ornithine transcarbamylase deficiency, cystic fibrosis\n\nQ: For which diseases is Intellia Therapeutics trying to develop treatments?\nA: transthyretin amyloidosis, hereditary angioedema, AATD-lung disease, hemophilia\n\nQ: For which diseases is Iovance Biotherapeutics trying to develop treatments?\nA: melanoma, cervical cancer, NSCLC\n\nQ: For which diseases is Ionis Pharmaceuticals trying to develop treatments?\nA: treatment-resistant hypertension, thrombotic disorders, IgA nephropathy, chronic heart failure with reduced ejection fraction, ATTR, CVD\n\nQ: For which diseases is Allogene trying to develop treatments?\nA: hematologic malignancies, solid tumors\n\nQ: For which diseases is "+stock_name+" trying to develop treatments?\n",
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["\n"]
    )
    answer = (str(response).split('"text": "A: ')[1]).split('"\n')[0]
    retval = str(answer)
    return retval

def get_pipeline(name):
    stock_name = name.replace(" ","+")
    print(stock_name)
    link = "https://clinicaltrials.gov/ct2/results/download_fields?down_count=10000&down_flds=shown&down_fmt=csv&recrs=abdf&lead="+stock_name+"&flds=b&flds=f&flds=c&flds=k&flds=p&flds=n&flds=y"
    r = requests.get(link)
    content = r.content
    text = str(content)
    rows = text.split("\\r\\n")
    trials = []
    for row in rows[1:]:
        trial = {}
        fields = row.split('","')
        if len(fields) > 2:
            try:
                trial["INTERVENTION"] = fields[4].split(": ")[1]
            except:
                trial["INTERVENTION"] = fields[4]
            if trial["INTERVENTION"].find("Non-Interventional") == -1:
                trial["NCT NUMBER"] = fields[0].split(',"')[1]
                for field in fields:
                    if len(field) > 200:
                        fields.remove(field)
                trial["PHASE"] = fields[6]
                try:
                    trial["EXPECTED COMPLETION"] = fields[7]
                except:
                    trial["EXPECTED COMPLETION"] = "UNKNOWN"
                try:
                    trial["TRIAL START"] = fields[8]
                except:
                    trial["TRIAL START"] = "UNKNOWN"
                trials.append(trial)
    print("PHASE, INTERVENTION, NCT NUMBER, EXPECTED COMPLETION, TRIAL START")
    for trial in trials:
        if trial["EXPECTED COMPLETION"].find("https://") != -1:
            trials.remove(trial)
        elif trial["PHASE"].find("Phase") == -1:
            trials.remove(trial)
        elif trial["TRIAL START"].find("https://") != -1:
            trials.remove(trial)
        elif trial["TRIAL START"].find("UNKNOWN") != -1:
            trials.remove(trial)
        elif trial["INTERVENTION"].find("89Zr\xcb\x97DFO\xcb\x97") != -1:
            trial["INTERVENTION"] = trial["INTERVENTION"].split("89Zr\xcb\x97DFO\xcb\x97")[1]
        elif trial["INTERVENTION"].find("|Drug") != -1:
            trial["INTERVENTION"] = trial["INTERVENTION"].replace("|Drug","")
        elif trial["TRIAL START"].find('",,"') != -1:
            trial["TRIAL START"] = trial["TRIAl START"].split('",,"')[0]
    for trial in trials:
        print(trial["PHASE"] + ", " + trial["INTERVENTION"] +", "+ trial["NCT NUMBER"] +", "+trial["EXPECTED COMPLETION"] +", "+ trial["TRIAL START"])
    return trials


st.title("Stock Analyzer v0.1")
name = st.sidebar.text_input("Name of Company")
analyze = st.sidebar.button("ANALYZYE")

if analyze:
    st.write("Fetching data for " + name + "...")
    openai.api_key = st.secrets["API_KEY"]
    website = get_website_name(name)
    ticker = get_ticker(name)
    industry = get_industry(name)
    st.write("FETCHING DATA FOR " + name.upper())
    st.write("WEBSITE: "+(str(website)))
    st.write("TICKER: "+(str(ticker)))
    st.write("INDUSTRY: "+(str(industry)))
    if industry.find("pharmaceutical") != -1:
        indications = get_diseases(name)
        st.write("INDICATIONS: " + (str(indications)))
        trials = get_pipeline(name)
        st.write("\n\n----FETCHING CLINICAL PIPELINE----")
        st.write("PHASE, INTERVENTION, NCT NUMBER, EXPECTED COMPLETION, TRIAL START")
        for trial in trials:
            st.write(trial["PHASE"] + ", " + trial["INTERVENTION"] +", "+ "https://clinicaltrials.gov/ct2/show/"+trial["NCT NUMBER"] +", "+trial["EXPECTED COMPLETION"] +", "+ trial["TRIAL START"])

