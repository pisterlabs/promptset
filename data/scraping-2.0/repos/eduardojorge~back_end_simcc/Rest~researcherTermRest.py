from flask import Flask, jsonify, request,Blueprint

import json
#import nltk
import unidecode
from flask_cors import CORS,cross_origin
import os
import Dao.termFlowSQL as termFlowSQL

from Model.Researcher import Researcher
from Model.Bibliographic_Production_Researcher import Bibliographic_Production_Researcher
from Model.Patent_Researcher import Patent_Researcher
from Model.Software_Researcher import Software_Researcher
from Model.Brand_Researcher import Brand_Researcher
from Model.Book_Researcher import Book_Researcher
from Model.Book_Chapter_Researcher import Book_Chapter_Researcher
from Model.Guidance_Researcher import Guidance_Researcher
from Model.Researcher_Report import Researcher_Report
from Model.Year_Barema import Year_Barema
from Model.PEvent_Researcher import PEvent_Researcher
import  Dao.resarcher_baremaSQL as  resarcher_baremaSQL
import Dao.areaFlowSQL


##from server import app

#https://www.fullstackpython.com/flask-json-jsonify-examples.html
#app = Flask(__name__)
#app.config["CORS_HEADERS"] = "Content-Type"
#app.route('/')
#CORS(app, resources={r"/*":{"origins":"*"}})
#app = Flask(__name__)

#if __name__ == '__main__': app.run(host='192.168.15.69',port=5000)
researcherTermRest = Blueprint('researcherTermRest', __name__)






######### Fluxo Termo

@researcherTermRest.route('/resarcher_barema', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def resarcher_barema():

    list_name= request.args.get('name')
   
    list_lattes_id = request.args.get('lattes_id')
    year = Year_Barema()
    
    year.article=request.args.get('yarticle')
    year.work_event=request.args.get('ywork_event')
 
    year.book=request.args.get('ybook')
    year.chapter_book=request.args.get('ychapter_book')
 

    year.patent=request.args.get('ypatent')
    year.software=request.args.get('ysoftware')
    year.brand=request.args.get('ybrand')
    year.resource_progress=request.args.get('yresource_progress')
    year.resource_completed=request.args.get('yresource_completed')
    year.participation_events=request.args.get('yparticipation_events')
      



    
 

    
  
    
    
    return jsonify(resarcher_baremaSQL.researcher_production_db(list_name,list_lattes_id,year)
), 200



@researcherTermRest.route('/originals_words', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def originals_words():
    list_researcher  = []
    initials = request.args.get('initials')
    type = request.args.get('type')
   # graduate_program_id =request.args.get('graduate_program_id')
    print(initials)
    ''''
     ''''''    
     filtergraduate_program=""
     if graduate_program_id!="":
       graduate_program_id = "AND gpr.graduate_program_id="+graduate_program_id
    
    if graduate_program_id is None:
        graduate_program_id =""
        
     '''   

    

    df_bd =termFlowSQL.list_research_dictionary_db(initials.lower(),type)

    list=[]
    for i,infos in df_bd.iterrows():
        research_dictionary  = {
        'term': str(infos.term),
        'frequency': str(infos.frequency),
        'type':str(infos.type),
        'checkbox':0
         }
        list.append( research_dictionary )
    
    return jsonify(list), 200


@researcherTermRest.route('/researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def research():
    list_researcher  = []

    terms = str(request.args.get('terms'))

    boolean_condition=request.args.get('boolean_condition')
    if boolean_condition is None:
       boolean_condition="or"
   
    if terms=="":
       return jsonify(list_researcher), 200
    university=""
    university = str(request.args.get('university'))+""
    #stemmer = nltk.RSLPStemmer()
    print(terms)
    termNovo=unidecode.unidecode(terms.lower())
    print(terms)
    type = request.args.get('type')

    graduate_program_id =request.args.get('graduate_program_id')
    #print("yyyyy "+graduate_program_id  )
    if graduate_program_id is None:
        graduate_program_id =""
    if graduate_program_id=="0":
        graduate_program_id =""

    #terms = unidecode(terms.lower())
    #print(termNovo)
   # print(stemmer.stem(termNovo))
    df_bd =termFlowSQL.list_researchers_originals_words_db(termNovo,university,type,boolean_condition,graduate_program_id)
    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        #area = Dao.areaFlowSQL.lists_great_area_expertise_researcher_db(infos.id)

        print(" qtd=" +str(infos.qtd))
       
       
        r = Researcher()
        r.id = str(infos.id)
        r.name = str(infos.researcher_name)
        r.among = str(infos.qtd)
        r.articles  =str(infos.articles)
        r.book_chapters =str(infos.book_chapters)
        r.book =str(infos.book)
        r.patent = str(infos.patent)
        r.software = str(infos.software)
        r.brand = str(infos.brand)
        r.university =str(infos.institution)
        r.lattes_id = str(infos.lattes)
        r.lattes_10_id =str(infos.lattes_10_id)
        r.abstract =str(infos.abstract)
        r.area =str(infos.area.replace("_"," "))
        r.city= str(infos.city)
        r.orcid =str(infos.orcid)
        r.image_university =str(infos.image)
        r.graduation = str(infos.graduation)
        r.lattes_update = str(infos.lattes_update)


        #print(r.getJson())
        
        researcher = r.getJson()
        
              
        #print(researcher)
        list_researcher.append(researcher) 
     


    return jsonify(list_researcher), 200

#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')
@researcherTermRest.route('/guidance_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def guidance_researcher():
    list_guidance_researcher  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
   

   

   
    df_bd =termFlowSQL.lists_guidance_researcher_db(researcher_id,1000)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        g = Guidance_Researcher()
        g.id =  str(infos.id)
        g.title = str(infos.title)
        g.nature = str(infos.nature)
        g.oriented= str(infos.oriented)
        g.type = str(infos.type)
        g.status = str(infos.status)
        g.year =  str(infos.year)
     
       


        #print(researcher)
        list_guidance_researcher.append(g.getJson()) 

    return jsonify(list_guidance_researcher), 200





#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')
@researcherTermRest.route('/brand_production_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def brand_production_researcher():
    list_brand_production_researcher  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
   

   

   
    df_bd =termFlowSQL.lists_brand_production_researcher_db(researcher_id,1000)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        b = Brand_Researcher()
        b.id =  str(infos.id)
        b.title = str(infos.title)
        b.year =  str(infos.year)
     
       


        #print(researcher)
        list_brand_production_researcher.append(b.getJson()) 

    return jsonify(list_brand_production_researcher), 200



#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')
@researcherTermRest.route('/book_production_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def book_production_researcher():
    list_book_production_researcher  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
    term= request.args.get('term')

   

   

   
    df_bd =termFlowSQL.lists_book_production_researcher_db(researcher_id,year,term)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        b = Book_Researcher()
        b.id =  str(infos.id)
        b.title = str(infos.title)
        b.year =  str(infos.year)
        b.isbn = str(infos.isbn)
        b.publishing_company = str(infos.publishing_company)
     
       


        #print(researcher)
        list_book_production_researcher.append(b.getJson()) 

    return jsonify(list_book_production_researcher), 200


#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')
@researcherTermRest.route('/book_chapter_production_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def book_chapter_production_researcher():
    list_book_chapter_production_researcher  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    term= request.args.get('term')
    year=request.args.get('year')
   

   

   
    df_bd =termFlowSQL.lists_book_chapter_production_researcher_db(researcher_id,year,term)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        b = Book_Chapter_Researcher()
        b.id =  str(infos.id)
        b.title = str(infos.title)
        b.year =  str(infos.year)
        b.isbn = str(infos.isbn)
        b.publishing_company = str(infos.publishing_company)
     
       


        #print(researcher)
        list_book_chapter_production_researcher.append(b.getJson()) 

    return jsonify(list_book_chapter_production_researcher), 200

#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')






@researcherTermRest.route('/researcher_report', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def researcher_report():
    list_researcher_report  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
   

   

   
    df_bd =termFlowSQL.lists_Researcher_Report_db(researcher_id,year)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        rr = Researcher_Report()
        rr.id =  str(infos.id)
        rr.title = str(infos.title)
        rr.year =  str(infos.year)
        rr.financing = str(infos.financing_institutionc)
        rr.project_name = str(infos.project_name)
     
       


        #print(researcher)
        list_researcher_report .append(rr.getJson()) 
     


 
    return jsonify(list_researcher_report), 200


@researcherTermRest.route('/software_production_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def software_production_researcher():
    list_software_production_researcher  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
   

   

   
    df_bd =termFlowSQL.lists_software_production_researcher_db(researcher_id,year)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        s = Software_Researcher()
        s.id =  str(infos.id)
        s.title = str(infos.title)
        s.year =  str(infos.year)
     
       


        #print(researcher)
        list_software_production_researcher.append(s.getJson()) 
     


 
    return jsonify(list_software_production_researcher), 200


@researcherTermRest.route('/pevent_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def pevent_researcher():
    list_pevent_researcher  = []
    #terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')

    term=request.args.get('term')

    nature=request.args.get('nature')
   

   

   
    df_bd =termFlowSQL.lists_pevent_researcher_db(researcher_id,year,term,nature)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        #print(str(infos.form_participation))

        p = PEvent_Researcher()
        p.id =  str(infos.id)
        p.event_name = str(infos.event_name)
        p.nature = str(infos.nature)
        p.participation = str(infos.form_participation)
        p.year =  str(infos.year)
       # print(p.participation)
     
       


        #print(researcher)
        list_pevent_researcher.append(p.getJson()) 
     


 
    return jsonify(list_pevent_researcher), 200

#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')
@researcherTermRest.route('/patent_production_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def patent_production_researcher():
    list_patent_production_researcher  = []
    #terms = request.args.get('terms')
    term= request.args.get('term')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
   

   

   
    df_bd =termFlowSQL.lists_patent_production_researcher_db(researcher_id,year,term)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        p = Patent_Researcher()
        p.id =  str(infos.id)
        p.title = str(infos.title)
        p.year =  str(infos.year)
        p.grant_date = str(infos.grant_date)
       


        #print(researcher)
        list_patent_production_researcher.append(p.getJson()) 
     


 
    return jsonify(list_patent_production_researcher), 200







#lists_bibliographic_production_article_researcher_db("Robótica",'35e6c140-7fbb-4298-b301-c5348725c467')
@researcherTermRest.route('/bibliographic_production_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def bibliographic_production_researcher():
    list_bibliographic_production_researcher  = []
    terms = request.args.get('terms')
    researcher_id = request.args.get('researcher_id')
    year=request.args.get('year')
    boolean_condition=request.args.get('boolean_condition')

    qualis = request.args.get('qualis')

    if boolean_condition is None:
       boolean_condition="or"
     
   

   

    #stemmer = nltk.RSLPStemmer()
  
    termNovo=unidecode.unidecode(terms.lower())
    type = request.args.get('type')
    print(boolean_condition)
 
    #terms = unidecode(terms.lower())
    #print(termNovo)
   # print(stemmer.stem(termNovo))
    df_bd =termFlowSQL.lists_bibliographic_production_article_researcher_db(termNovo,researcher_id,year,type,boolean_condition,qualis)

    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():

        b = Bibliographic_Production_Researcher()
        b.id =  str(infos.id)
        b.title = str(infos.title)
        b.year =  str(infos.year)
        b.type = str(infos.type)
        b.doi = str(infos.doi)
        b.qualis = str(infos.qualis)
        b.magazine = str(infos.magazine)
        b.researcher = str(infos.researcher)
        b.lattes_10_id = str(infos.lattes_10_id)
        b.lattes_id = str(infos.lattes_id)
        b.jif = str(infos.jif)
        b.jcr_link =str(infos.jcr_link)


        #print(researcher)
        list_bibliographic_production_researcher.append(b.getJson()) 
     


 
    return jsonify(list_bibliographic_production_researcher), 200









@researcherTermRest.route('/qualis_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def qualis_researcher():
    
    researcher_id = request.args.get('researcher_id')
    year = request.args.get('year')
    lists_qualis=[]
    
    #terms = unidecode(terms.lower())
    #print(termNovo)
   # print(stemmer.stem(termNovo))
    graduate_program_id =request.args.get('graduate_program_id')
    if graduate_program_id is None:
        graduate_program_id =""
    df_bd =termFlowSQL.lists_bibliographic_production_qtd_qualis_researcher_db(researcher_id,year,graduate_program_id)

   
    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():
        qualis  = {
       

        'among': str(infos.qtd),
      
        'qualis':str(infos.qualis)
       # 'year':str(infos.year)

        }
        #print(researcher)
        lists_qualis.append(qualis) 

    return jsonify(lists_qualis), 200






@researcherTermRest.route('/lists_word_researcher', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def lists_word_researcher():
    
    researcher_id = request.args.get('researcher_id')
    
     

    graduate_program_id =request.args.get('graduate_program_id')
    if graduate_program_id is None:
        graduate_program_id =""

    lists_word=[]
    
    #terms = unidecode(terms.lower())
    #print(termNovo)
   # print(stemmer.stem(termNovo))
    df_bd =termFlowSQL.lists_word_researcher_db(researcher_id,graduate_program_id)

    'qtd','term'
    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():
        words  = {
       

        'among': str(infos.qtd),
        'term':str(infos.term)
       

        }
        #print(researcher)
        lists_word.append(words) 

    return jsonify(lists_word), 200


@researcherTermRest.route('/institutionFrequenci', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def institutionFrequenci():
    list_institutionFrequenci  = []
    terms = request.args.get('terms')
    #stemmer = nltk.RSLPStemmer()
    print(terms)
    termNovo=terms.lower()
    print(terms)
    university = str(request.args.get('university'))+""

    type_ = str(request.args.get('type'))+""
    #terms = unidecode(terms.lower())
    #print(termNovo)
   # print(stemmer.stem(termNovo))
    #type_="ARTICLE"
    #print(type_)
    df_bd =termFlowSQL.lista_institution_production_db(termNovo,university,type_)
    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():
        institution  = {
        'id': str(infos.id),
        'institution': str(infos.institution),
        'among': str(infos.qtd),
        'image':str(infos.image)
       

        }
        #print(researcher)
        list_institutionFrequenci.append(institution) 

    return jsonify(list_institutionFrequenci), 200


#print(list_researchers_originals_words_db("robótica | robô | robotics"))   
@researcherTermRest.route('/researcherID', methods=['GET'])
@cross_origin(origin="*", headers=["Content-Type"])
def researcherID():
    list_researcher  = []
    researcher_id = request.args.get('researcher_id')
  
    #termNovo=unidecode.unidecode(name.replace(";","&"))
   
    #print(termNovo)
   # print(stemmer.stem(termNovo))
    #df_bd =SimccBD.lista_researcher_name_db(name.lower())
    df_bd =termFlowSQL.lista_researcher_id_db(researcher_id)
    #df_bd.sort_values(by="articles", ascending=False, inplace=True)
    for i,infos in df_bd.iterrows():
        #area = Dao.areaFlowSQL.lists_great_area_expertise_researcher_db(infos.id)

        r = Researcher()
        r.id = str(infos.id)
        r.name = str(infos.researcher_name)
        r.among = str("0")
        r.articles  =str(infos.articles)
        r.book_chapters =str(infos.book_chapters)
        r.book =str(infos.book)
        r.patent = str(infos.patent)
        r.software = str(infos.software)
        r.brand = str(infos.brand)
        r.university =str(infos.institution)
        r.lattes_id = str(infos.lattes)
        r.lattes_10_id =str(infos.lattes_10_id)
        r.abstract =str(infos.abstract)
        r.area =str(infos.area.replace("_"," "))
        r.city= str(infos.city)
        r.orcid =str(infos.orcid)
        r.image_university =str(infos.image)
        r.graduation = str(infos.graduation)
        r.lattes_update = str(infos.lattes_update)


        #print(r.getJson())
        
        researcher = r.getJson()

        #print(researcher)
        list_researcher.append(researcher) 
    return jsonify(list_researcher), 200




##############################################################################

if __name__ == '__main__':
    # run app in debug mode on port 5000
    researcherTermRest.run(debug=True, port=5001, host='0.0.0.0')