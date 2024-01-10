# -*- coding: utf-8 -*-

import json
from datetime import datetime
import io
import jsonlines
import random
from fpdf import FPDF
import requests
from html_to_etree import parse_html_bytes
from extract_social_media import find_links_tree
from extract_emails import EmailExtractor
from extract_emails.browsers import RequestsBrowser
import openai
import whois


# open the files in universal line ending mode 
file1 = open('words.txt', 'r')
list1 = list(file1.read().split())
    
file2 = open('status.txt', 'r')
list2 = list(file2.read().split())    

file11 = open('words2.txt', 'r')
list11 = list(file11.read().split())
del list11[0]  
  
file22 = open('status2.txt', 'r')
list22 = list(file22.read().split())    
del list22[0]  

file3 = open('url.txt', 'r')
url = file3.read().split()
url = url[0]

print(url)

#Read SQLi payloads database
with open('docs/SQLIPayloads.txt', encoding="utf-8") as f:
    SQLIPayloads = f.readlines()


myList=list1+list11
suspectedTypes = []
aisummary = []
#Current method is GET (default dirsearch setting)
method = 'GET'


#Get suspected file types
item = ".php"
if  True in list(map(lambda el : item in el ,myList)):
    print(item)
    suspectedTypes.append(item)
    
item = ".html"
if  True in list(map(lambda el : item in el ,myList)):
    print(item)
    suspectedTypes.append(item)    

item = ".aspx"
if  True in list(map(lambda el : item in el ,myList)):
    print(item)
    suspectedTypes.append(item)    

item = ".js"
if  True in list(map(lambda el : item in el ,myList)):
    print(item)
    suspectedTypes.append(item)
    
item = ".jsp"
if  True in list(map(lambda el : item in el ,myList)):
    print(item)
    suspectedTypes.append(item)    

paths = []
p = list1 + list11

status = []
s = list2 + list22


#delete duplicates
for i in range(len(p)):
    if p[i] not in paths:
        paths.append(p[i])
        status.append(s[i])


#Fetch current time
now = datetime.now().replace(microsecond=0)
timestamp = int(datetime.timestamp(now))

#create metadata file name
filename = str(timestamp)+"cheatsheet.jsonl"

#Print to console
print(paths)
print(status)
print(filename)


#social media extraction
res = requests.get(url)
tree = parse_html_bytes(res.content, res.headers.get('content-type'))

links = list(find_links_tree(tree))


#emails extraction
with RequestsBrowser() as browser:
    email_extractor = EmailExtractor(url, browser, depth=2)
    emails = email_extractor.get_emails()


#Define class for JSONLINES objects creation
class Info:
    
    def __init__(self, web_path, status_code, attacks):
        
        self.path = web_path
        self.status = status_code
        self.attacks = attacks
        
    def toJson(self):
        d = {}
        d['path'] = self.path
        d['status'] = self.status
        d['SQLIattacks'] = [attack for attack in self.attacks]
        return d

  
def store_metadata():   
    
    #populate JSONLINES objects    
    for i in range(len(paths)-1):
        
        web_path = url + paths[i+1]
        status_code = status[i+1]       
        
        #Define lists
        infos = []                                        
        sqli = []
        
        #Create entry   
        for i in range(5):
            random_index = random.randint(0,len(SQLIPayloads)-1)  
            sqli.append(SQLIPayloads[random_index])
            attacks = sqli
            
        
                    
        info = Info(web_path, status_code, attacks)
        infos.append(info.toJson())
                           
        #Append object to file
        with io.open(str(filename), 'a', encoding="utf-8") as file:
                 
            file.write(str(json.dumps(infos[-1])))
            file.write('\n')                                                                         
                    
    
    #Get text for AI summarization    
    with io.open('htmltext.txt') as f:
        
        htmltext = f.read()
                
        
    #OpenAI API    
    openai.api_key = "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"
    
    response = openai.Completion.create(
      engine="davinci",
      prompt=htmltext,
      temperature=0.3,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["++"]
    )
    
    print(htmltext)
    print(response)
    
    print(response,  file=open('htmltext.json', 'w'))
    
    with open('htmltext.json') as f:
        data = json.load(f)        
    
    aisummary.append(data['choices'][0]['text'])
    print(aisummary)   


#Create PDF layout        
def pdf():
    
    placeholderPath = []
    placeholderStatus = []
    
    #open jsonl file
    with jsonlines.open(str(filename)) as f:
        #read each line
        for line in f.iter():
    
            print('Path:',line['path'])
            placeholderPath.append(line['path'])
            placeholderStatus.append(line['status'])
            
     
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, 'Cheatsheet for '+ url, 1, 1, 'C')
    
    w = whois.whois(url)
    w.expiration_date  # dates converted to datetime object

    pdf.cell(190, 10, '**OSINT Information**', 1, 1, 'C')

        
    if w.text == '':
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(0, 10, 'No OSINT info available', 1, 'L')
    else:
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(0, 10, w.text, 1, 'L')
    
        
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, '**AI Powered Summary**', 1, 1, 'C')
    pdf.set_font('Arial', '', 16)
    pdf.multi_cell(190, 10, str(aisummary)[1:-1], 1, 1, 'C')
    
    
    
    pdf.set_font('Arial', 'B', 16)    
    pdf.cell(190, 10, '**Found Social Media Links**', 1, 1, 'C')
    
    if not links:
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(0, 10, 'No Social Media info available', 1, 'L')    
        print('No Links')	
    for i in range(len(links)):
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(0, 10, links[i], 1, 'L')    
    
    
    for email in emails:
        #print(email)
        print(email.as_dict()) 
    
    #catch exception if no emails are found    
    try:
        dictionary = email.as_dict()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, '**Found Emails**', 1, 1, 'C')
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(0, 10, dictionary["email"]+ ' -> ' + ' Source: ' 
                       + dictionary["source_page"], 1, 'L')
    
    except UnboundLocalError:
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(190, 10, '**Found Emails**', 1, 1, 'C')
        pdf.set_font('Arial', '', 16)
        pdf.multi_cell(0, 10, 'No Email Addresses available', 1, 'L')
        print("No Emails")
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, '**Suspected file types**', 1, 1, 'C')
    pdf.set_font('Arial', '', 16)
    pdf.cell(190, 10, str(suspectedTypes)[1:-1], 1, 1, 'C')

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(190, 10, '**Found Paths**', 1, 1, 'C')
    pdf.set_font('Arial', '', 16)
    pdf.cell(190, 10, 'Method: '+method+ ', Total #: '+ str(len(placeholderPath)), 1, 1, 'C')

    toPayloads = pdf.add_link()
    pdf.set_link(toPayloads, page=2)


    for i in range(len(placeholderPath)):
        if int(placeholderStatus[i]) == 200:
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(10, 10, str(i+1), 1, 0, 'C')
            pdf.set_font('Arial', '', 16)
            pdf.multi_cell(0, 10, placeholderPath[i] + '  ->  ' +
                           placeholderStatus[i] + '  ->  ' 
                           + 'SQLi Suggestions:', 1, 'L')
            
            for i in range(5):
                #pdf.set_font('Arial', '', 10)
                random_index = random.randint(0,len(SQLIPayloads)-1)
                
               
                pdf.set_font('ZapfDingbats', 'B', 8)
                pdf.cell(5, 10, '4', 1, 0, 'T')
                pdf.cell(5, 10, '5', 1, 0, 'T')
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 10,SQLIPayloads[random_index], 1, 'R')
                pdf.set_font('Arial', 'B', 16)
        else:
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(10, 10, str(i+1), 1, 0, 'C')
            pdf.set_font('Arial', '', 16)
            pdf.multi_cell(0, 10, placeholderPath[i] + '  ->  ' +
                           placeholderStatus[i], 1, 'L') 

    #Output PDF file    
    pdf.output('cheatsheet.pdf', 'F')


#Driver Code

#Call the store_metadata function
store_metadata()   
pdf()


print(links)   
    
    