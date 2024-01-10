from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from datetime import date
import sys
import os

tgrwaURL = 'https://mybidmatch.outreachsystems.com/go?sub=0F7D0F07-33E5-427C-95F8-8C7F014C8924'

# reportNum: position of myBidMatch report to query (starts at 1 with most recent)
# rfpRange: list of rfp's to probe within the report does not need to be ordered
def getRFPReport(reportNum, rfpNumStart, rfpNumEnd):
    print('Scraping Text!!!')
    service = Service()
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=service, options=options)
    # Open webPage from myBidMatch
    driver.get(tgrwaURL)
    # get first entry in table of reports
    reportTable = driver.find_element(By.CSS_SELECTOR,'body > table.data')
    myBidMatchReport = reportTable.find_element(By.CSS_SELECTOR,
                                              'tbody > tr:nth-child('+str(reportNum)+') > td:nth-child(1) > a')
    tabText = myBidMatchReport.text
    print(tabText)

    numRFPs = reportTable.find_element(By.CSS_SELECTOR,
                                       'tbody > tr:nth-child('+str(reportNum)+') > td:nth-child(2)').text

    if numRFPs == '0':
        print("No opportunities for this report!!!")
        driver.quit()
        return False
    else:
        # click on the table
        myBidMatchReport.click()

        rfpText = []
        # if no range specified do all rfps
        print("Setting rfpRange")
        # BPM: Capping number for now to avoid timeOut
        # Case1: range is longer than number of RFPs
        if int(numRFPs) < (rfpNumEnd - rfpNumStart) + 1:
            rfpRange = [iRFP for iRFP in range(1,int(numRFPs)+1)]
        elif (rfpNumEnd - rfpNumStart) > 30:
            # case 2: range is bigger than 30, cap it at 30
            rfpRange = [iRFP for iRFP in range(rfpNumStart,rfpNumStart+31)]
        else:
            # case 3: range works as is
            rfpRange = [iRFP for iRFP in range(rfpNumStart,rfpNumEnd+1)]

        for rfpNum in rfpRange:
            rfpTitle, rfpBody, rfpKeyWord, rfpLink = scrapeRFP(driver, rfpNum)
            rfpText.append([rfpTitle, rfpBody, rfpKeyWord, rfpLink])
            # need to go back to report page
            driver.back()

        driver.quit()
        print("Done textScraping!!!")
        return rfpText

def scrapeRFP(driver, rfpNum):
    # get desired entry of table of RFPs
    rfpTable = driver.find_element(By.CSS_SELECTOR,'body > table.data')
    RFPEnt = rfpTable.find_element(By.CSS_SELECTOR,
                                        'tbody > tr:nth-child('+str(rfpNum)+') > td:nth-child(5) > a')
    rfpText = RFPEnt.text
    print(rfpText)

    # Click on the first rfp link
    RFPEnt.click()

    # Get link to rfp
    rfpLink = driver.current_url

    # get to rfp text
    rfpText = driver.find_element(By.CSS_SELECTOR, 'body > div.art-box')

    # Grab Full title of RFP
    rfpTitle = rfpText.find_element(By.CSS_SELECTOR, 'h4').text
    # Grab rfpText
    # NOTE: Top disclaimer text is also name <p>,
    # so I need to use full path here. That is annoying
    rfpBodyElems = driver.find_elements(By.CSS_SELECTOR, 'body > div.art-box > p')
    rfpBody = ''
    for iRfp in rfpBodyElems:
        rfpBody += iRfp.text
    # grab rfp keywords
    rfpKeyWordElem = rfpText.find_elements(By.CSS_SELECTOR, 'i')
    rfpKeyWord = []
    for iRfp in rfpKeyWordElem:
        rfpKeyWord.append(iRfp.text)

    print("Done with this RFP!!!")
    return rfpTitle, rfpBody, rfpKeyWord, rfpLink

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY"),
)

def callGPT(prompt, model):
    today = date.today()
    d1 = today.strftime("%m/%d/%Y")
    sysPrompt = "You are a helpful assisstant. The current date is: "
    sysPrompt += d1
    response = client.chat.completions.create(
    messages=[
        {
          "role": "system",
          "content": sysPrompt,
        },
        {
          "role": "user",
          "content": prompt,
        }
    ],
    model=model,
    )
    return response

# taken from openai.com on 11/17/2023
costInDic = {
      'gpt-4' : 0.03/1000,
      'gpt-4-1106-preview' : 0.01/1000,
      'gpt-3.5-turbo' : 0.0015/1000,
      'gpt-3.5-turbo-1106' : 0.0010/1000
}
costOutDic = {
      'gpt-4' : 0.06/1000,
      'gpt-4-1106-preview' : 0.03/1000,
      'gpt-3.5-turbo' : 0.002/1000,
      'gpt-3.5-turbo-1106' : 0.0020/1000
}

def calcCost(tokens, isPrompt, model):
    if isPrompt:
      costPerToken = costInDic[model]
    else:
      costPerToken = costOutDic[model]
    return costPerToken*tokens


from pylatex import Document, Section, Hyperref, Package
from pylatex.utils import escape_latex, NoEscape

def hyperlink(url,text):
    text = escape_latex(text)
    return NoEscape(r'\href{' + url + '}{' + text + '}')

def writeOutput(doc, rfpTitle, output, rfpLink=None):
    with doc.create(Section(rfpTitle)):
        doc.append(output)
        if (rfpLink != None):
            doc.append("\n")
            doc.append("\n")
            doc.append(hyperlink(rfpLink, "Link to RFP"))

def probeRFPs(rfpTexts, basePrompt, model):
    # make latex doc to store output
    print("Probing RFPs")
    geometryOptions = {"tmargin" : "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options = geometryOptions)
    doc.packages.append(Package('hyperref'))
    allRFPCost = 0.0
    for iRFP in rfpTexts:
        title = iRFP[0]
        body = iRFP[1]
        keyWords = iRFP[2][-1]
        link = iRFP[3]
        print(title)
        # print(keyWords)

        promptEnd = """\nThe RFP text is given below between
        the <<< >>> brackets

        <<<\n""" + body + "\n>>>"
        prompt = basePrompt + promptEnd
        response = callGPT(prompt, model)
        out = response.choices[0].message.content
        tokens = response.usage
        print("TokensIn: ", tokens.prompt_tokens)
        promptCost = calcCost(tokens.prompt_tokens, True, model)
        print("CostIn: ", promptCost)
        print("TokensOut: ", tokens.completion_tokens)
        responseCost = calcCost(tokens.completion_tokens, False, model)
        print("CostOut: ", responseCost)
        print('totalCost = ', promptCost + responseCost)
        allRFPCost += promptCost + responseCost
        print("Writing to output")
        writeOutput(doc, title, out, link)
    # add prompt to end of pdf
    writeOutput(doc, "Prompt", basePrompt)
    writeOutput(doc, "Model", model)
    # generate pdf output
    doc.generate_pdf('/home/brycepm2/ZAMAutoWebsite/assets/pdfOut/RFPSummary', clean_tex=False)
    print('Cost for all RFPs: ', allRFPCost)
    print('Done with analysis!!!')
    return allRFPCost

def main():
    old_stdout = sys.stdout
    logFile = open("./RFPRun_Console.log", 'w')
    sTime = time.time()
    sys.stdout = logFile
    rfpStart = 42
    rfpEnd = 88
    rfpText = getRFPReport(3, rfpStart, rfpEnd)

    model = 'gpt-4-1106-preview'

    basePrompt = """You are a structural engineer assessing
    requests for proposals for your company to apply to.
    Your company specializes in building design and
    rehabilitation.

    Briefly summarize the following RFP and
    make a reccomendation on whether it is a good fit
    for the company.
    """
    probeRFPs(rfpText, basePrompt, model)
    eTime = time.time()
    print("Total time = ", (eTime - sTime))


if __name__ == '__main__':
    inputs = sys.argv
    main()
