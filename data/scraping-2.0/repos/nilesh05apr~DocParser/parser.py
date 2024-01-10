import urllib
import warnings
from pathlib import Path as p

import pandas as pd
import re
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter



class Retrive:
  def __init__(self, filepath):
    loader = PyPDFLoader(filepath)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    st = ""
    for i in texts:
      st += i
    self.st=st

    sents = st.split("\n")
    keys = ['vendor ','date for completion', 'land (address', 'plan details and', 'VACANT',"purchaser",'improvements','inclusion','exclusions','Land tax']

    dc = {}
    for item in keys:
        for x in sents:
            if item in x:
                print(item, x)
                dc[item] = [sents[sents.index(x)], sents[sents.index(x) + 1], sents[sents.index(x) + 2], sents[sents.index(x) + 3], sents[sents.index(x) + 4]]
                break  # Stop iterating after finding the first occurrence

# Now, dc will only contain values for keys based on the first occurrence in sentences.


    dc[keys[0]] = dc[keys[0]][0] #vendor's name
    dc[keys[1]] = dc[keys[1]][0] #date of completion
    dc['VACANT'] = dc['VACANT'][:-2]
    dc[keys[5]] = dc[keys[5]][-2]
    dc['improvements'] = dc['improvements'][0:2]
    dc['land (address']=dc['land (address'][2:4]
    dc['exclusions']=dc['exclusions'][0]
    dc['Land tax']=dc['Land tax'][0]

    self.dc = dc


  def getName(self):
    return self.dc['vendor '].split(' ',1)[1]

  def getLandaddress(self):
    return self.dc['land (address'][0].split(')',1)[1]

  def getPlandetails(self):
    return self.dc['land (address'][1].split(':',1)[-1]

  def getSettlementdate(self):
    doc = self.dc['date for completion'].split('(',1)
    if len(doc) == 0:
      return "Need to be confirmed"
    return doc[0].split('date for completion')[1]

  def getLandstatus(self):
    sts = re.findall(r'[\uf0fe☒]\s*([^☐\uf0fe]+)', self.dc['VACANT'][0])
    if len(sts) == 0:
      return "Further clarification is also required of whether the Vendor will be able to provide vacant possession on settlement."
    return sts

  def getPrice(self):
    pattern = re.compile(r'\bbalance\b\s*(\$[\d,.]+)')
    matches = pattern.findall(self.st)
    print(matches)
    #price = dc['purchaser’s solicitor    '].split('balance')[1].strip()
    if len(matches) == 0:
      return "TBA"
    return matches[0]

  def getImprovments(self):
    selected_options = re.findall(r'(?:X|\uf0fe)\s*([^X\s]+)', self.dc['improvements'][0])

    if len(selected_options) == 0:
      return "Need to be confirmed"
    elif len(selected_options) == 1:
      return selected_options[0]
    else:
      impvs = ""
      for i in range(0, len(selected_options)-1):
        impvs += selected_options[i] + ','
      impvs += 'and' + selected_options[-1]
      return impvs

  def getInclusions(self):
    inc = re.findall(r'\uf0fe\s*([^☐\uf0fe]+)', self.dc['inclusion'][0])

    if len(inc) != 0:
      return "Inclusions are marked under the inclusion tab of the contract"
    return "Inclusions are not marked under the inclusion tab of the contract"

  def getExclusions(self):
    exc = self.dc['exclusions'].split('exclusions')[1]
    return exc

  def getLandtax(self):
    ltx = re.findall(r'\uf0fe\s*([^☐\uf0fe]+)', self.dc['Land tax'])
    if len(ltx) == 0:
      return "Land tax is not marked as adjustable or not adjustable"
    elif ltx[0].strip().lower() == 'no':
      return "Land tax is marked as not adjustable"
    else:
      return "Land tax is marked as adjustable"
