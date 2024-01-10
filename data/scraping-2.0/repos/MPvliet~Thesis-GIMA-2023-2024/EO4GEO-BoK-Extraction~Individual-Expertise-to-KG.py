import json, time, re, uuid, os
from rdflib import URIRef, Literal, BNode, Namespace, Dataset
from rdflib.collection import Collection
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
  start_time = time.time()
  #processAllInputJSON()
  parseIndividualExpertiseIntoRDF()
  duration = time.time() - start_time
  print("Scripte duurde: ", duration, "seconds")

def parseIndividualExpertiseIntoRDF():
  ds = Dataset()
  g = ds.graph(identifier=URIRef("https://bok.eo4geo.eu/applications"))

  # define ontologies that are used.
  obok = Namespace('http://example.org/OBOK/')
  boka = Namespace('http://example.org/BOKA/')
  dce = Namespace('http://purl.org/dc/elements/1.1/')
  org = Namespace('http://www.w3.org/ns/org#')
  rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
  bibo = Namespace('http://purl.org/ontology/bibo/')
  foaf = Namespace('http://xmlns.com/foaf/0.1/')
  rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
  schema = Namespace('https://schema.org/')
  dcterms = Namespace('http://purl.org/dc/terms/')
  skos = Namespace('http://www.w3.org/2004/02/skos/core#')
  eo4geo = Namespace('https://bok.eo4geo.eu/')

  # bind ontologies, otherwise the prefixes are not correct. 
  g.bind('obok', obok)
  g.bind('boka', boka)
  g.bind('dce', dce)
  g.bind('org', org)
  g.bind('rdf', rdf)
  g.bind('bibo', bibo)
  g.bind('foaf', foaf)
  g.bind('rdfs', rdfs)
  g.bind('schema', schema)
  g.bind('dcterms', dcterms)
  g.bind('skos', skos)
  g.bind('eo4geo', eo4geo)

  with open('EO4GEO-BoK-Extraction\input\EO4GEOBOKDATA.json', 'r') as f:
    data = json.load(f)

  # creates a dictionary including the concept notatation and the full name. The full name is received from the NLP ouptut. But needs to be matched with the notation ex. WB4 
  conceptDict = {} 
  for eo4geoConcept in data:
    conceptDict[data[eo4geoConcept]['name']] = {
      'id': data[eo4geoConcept]['id']
    }

  # opens 
  with open ('EO4GEO-BoK-Extraction\input\IndividualExpertise.json', 'r') as file:
    indivudalExpertise = json.load(file)
  
  uniqueOrganisationDict = {}
  uniqueAuthorDict = {}

  #Create uniqueAuthorDict with all necessarily information to make RDF triples
  for expertise in indivudalExpertise:
    #print(expertise['doi'])
    
    for author in expertise['authors']:
      if re.match(r'^-?\d+(\.\d+)?$', author[-1]):
        authorName = re.match(r'^(.*?)\d', author).group(1).strip().rstrip()
        authorOrgNumberList = re.findall(r'\d+', author)
        authorOrgNumberListStr = [str(number) for number in authorOrgNumberList]

        if authorName not in uniqueAuthorDict:
          uniqueAuthorDict[authorName] = {
            'authorName': authorName,
            'authorURI': str(uuid.uuid4()),
            'worksAt': []
          }

        for organisation in expertise['organisations']:
          if any(organisation.startswith(num) for num in authorOrgNumberListStr):
            uniqueAuthorDict[authorName]['worksAt'].append({'organisationName': organisation[1:].strip().rstrip()})
          if re.match(r'^-?\d+(\.\d+)?$', organisation[0]):
            if organisation[0] not in uniqueOrganisationDict:
              uniqueOrganisationDict[organisation[1:].strip().rstrip()] = { # strip() removes whitespace at the begin of the line rstrip()at the end
                'organisationName': organisation[1:].strip().rstrip(),
                'organisationURI': str(uuid.uuid4())
              }
      else:
        authorName = author.strip().rstrip()
        authorOrgNumberListStr = ['1']
        if authorName not in uniqueAuthorDict:
          uniqueAuthorDict[authorName] = {
            'authorName': authorName,
            'authorURI': str(uuid.uuid4()),
            'worksAt': []
          }
        
        for organisation in expertise['organisations']:
          uniqueAuthorDict[authorName]['worksAt'].append({'organisationName': organisation.strip().rstrip()})
          if organisation not in uniqueOrganisationDict:
            uniqueOrganisationDict[organisation.strip().rstrip()] = {
              'organisationName': organisation.strip().rstrip(),
              'organisationURI': str(uuid.uuid4())
            }
        
    for author in uniqueAuthorDict:
      for orgDict in uniqueAuthorDict[author]['worksAt']:
        if orgDict['organisationName'] in uniqueOrganisationDict:
          orgDict['organisationURI'] = uniqueOrganisationDict[orgDict['organisationName']]['organisationURI']
    
    #create document class
    doi = expertise['doi']
    doiURI = URIRef('{}'.format(doi))
    g.add((doiURI, rdf.type, bibo.Report))
    g.add((doiURI, bibo.doi, Literal(doi)))

    for concept in expertise['concepts']:
      conceptURI = URIRef('{}{}'.format('https://bok.eo4geo.eu/', conceptDict[concept]['id']))

      #link document to EO4GEO concept
      g.add((conceptURI, boka.describedIn, doiURI))
      
      # create hasknowledeOf and personWithKNowledge constructs
      for author in expertise['authors']:
        if re.match(r'^-?\d+(\.\d+)?$', author[-1]):
          authorName = re.match(r'^(.*?)\d', author).group(1).strip().rstrip()
        else:
          authorName = author.strip().rstrip()
        expertURI = URIRef('{}{}'.format('https://bok.eo4geo.eu/',uniqueAuthorDict[authorName]['authorURI']))
        g.add((expertURI, boka.hasKnowledgeOf, conceptURI))
        g.add((conceptURI, boka.personWithKnowledge, expertURI))
        g.add((expertURI, boka.authorOf, doiURI))
        
    # Creates the bibo:authorList relation
    collection_node = BNode()
    authorsList = []
    for author in expertise['authors']:
      if re.match(r'^-?\d+(\.\d+)?$', author[-1]):
          authorName = re.match(r'^(.*?)\d', author).group(1).strip().rstrip()
      else:
        authorName = author.strip().rstrip()
      expertURI = URIRef('{}{}'.format('https://bok.eo4geo.eu/',uniqueAuthorDict[authorName]['authorURI']))
      authorsList.append(expertURI)


    collection = Collection(g, collection_node, authorsList)
    g.add((doiURI, bibo.authorList, collection_node))

  #create Expert class and Organisation class
  for author in uniqueAuthorDict:
    expertURI = URIRef('{}{}'.format('https://bok.eo4geo.eu/',uniqueAuthorDict[author]['authorURI']))
    g.add((expertURI, rdf.type, boka.Expert))
    g.add((expertURI, foaf.name, Literal(uniqueAuthorDict[author]['authorName'])))
    g.add((expertURI, rdfs.label, Literal(uniqueAuthorDict[author]['authorName'])))
    

    for organisation in uniqueAuthorDict[author]['worksAt']:
      organisationURI = URIRef('{}{}'.format('https://bok.eo4geo.eu/',organisation['organisationURI']))
      g.add((organisationURI, rdf.type, org.Organization))
      g.add((expertURI, org.memberOf, organisationURI))
      g.add((organisationURI, org.hasMember, expertURI))
      g.add((organisationURI, foaf.name, Literal(organisation['organisationName'])))
      g.add((organisationURI, rdfs.label, Literal(organisation['organisationName'])))

  g.serialize(destination="EO4GEO-BoK-Extraction\output\EO4GEO-KG-individual.trig", format="trig")

def processIndivudalExpertiseJson(jsonPrompt):
  client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
  )

  #jsonPrompt = '{"DOI": "https://doi.org/10.5194/agile-giss-4-18-2023", "Title": "Predicting Pedestrian Counts using Machine Learning Molly Asher1 , Yannick Oswald1 , and Nick Malleson1 1School of Geography , University of Leeds , UK Correspondence : Nick Malleson ( n.s.malleson @ leeds.ac.uk )","Concepts": ["Discovery over linked open data","Open data","Machine learning","Approaches to point, line, and area generalization","Publishing linked open data","Decision trees","Time","Information-as-data-interpretation"]}'

  expectedJSONResult = '{"doi": "https://doi.org/10.5194/agile-giss-4-20-2023","authors": ["Nick Bearman1","Rongbo Xu2","Patrick J. Roddy3","James D. Gaboardi4","Qunshan Zhao5","Huanfa Chen6","Levi Wolf7"],"organisations": ["1Geospatial Training Solutions and University College London, London, UK","2Urban Big Data Centre, University of Glasgow, Glasgow, UK","3Advanced Research Computing, University College London, London, UK","4Geospatial Science and Human Security Division, Oak Ridge National Laboratory, USA","5Urban Big Data Centre, University of Glasgow, Glasgow, UK","6CASA, University College London, London, UK","7University of Bristol, Bristol, UK"],"concepts": ["Time","Information-as-data-interpretation"]}'

  messages = [
    {"role": "system", "content": 'You can help me parse a single JSON I will provide, in the following JSON structure: `[{"doi": "","authors": [],"organisations": [],"concepts": []}] only return the json and you can keep the numbers before the organisation and behind each authorsname and remove the "and" before the last author name. Also make sure to enclose property names in the json with double quotes. At last double check if the created output is indeed the specified format, othwerise adjust. Thanks!'},
    {"role": "user", "content": ' for example this json {"DOI": "https://doi.org/10.5194/agile-giss-4-20-2023","Title": "Developing capacitated p median location allocation model in the spopt library to allow UCL student teacher placements using public transport Nick Bearman \ufffd1 , Rongbo Xu2 , Patrick J. Roddy \ufffd3 , James D. Gaboardi \ufffd4 , Qunshan Zhao \ufffd5 , Huanfa Chen \ufffd6 , and Levi Wolf \ufffd7 1Geospatial Training Solutions and University College London , London , UK 2Urban Big Data Centre , University of Glasgow , Glasgow , UK 3Advanced Research Computing , University College London , London , UK 4Geospatial Science and Human Security Division , Oak Ridge National Laboratory , USA 5Urban Big Data Centre , University of Glasgow , Glasgow , UK 6CASA , University College London , London , UK 7University of Bristol , Bristol , UK Correspondence : Nick Bearman ( nick @ nickbearman.com )", "Concepts": ["Time","Information-as-data-interpretation"]}'},
    {"role": "assistant", "content": expectedJSONResult},
    {"role": "user", "content": '{}'.format(jsonPrompt)}
  ]

  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  print(response.choices[0].message.content)
  return response.choices[0].message.content


def mergeToTotalExpertiseFile(newParsedData):
  with open('EO4GEO-BoK-Extraction\input\IndividualExpertise.json', 'r') as file:
    existingData = json.load(file)
  
  # Append the new data
  existingData.append(newParsedData)
  #existingData.extend(newParsesdData)

  with open('EO4GEO-BoK-Extraction\input\IndividualExpertise.json', 'w') as file:
    json.dump(existingData, file)

def processAllInputJSON():
  indivudalExpertiseFolder = 'EO4GEO-BoK-Extraction\input\Individual-Expertise'
  for file in os.listdir(indivudalExpertiseFolder):
    file_path = os.path.join(indivudalExpertiseFolder, file)
    if os.path.isfile(file_path):
      with open (file_path, 'r') as file:
        jsonAsText = file.read()
      parsedData = processIndivudalExpertiseJson(json.loads(jsonAsText)) #let openai parse the json in a better json structure
      newJson = json.loads(parsedData)
      mergeToTotalExpertiseFile(newJson)

main()