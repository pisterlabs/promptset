from openai.embeddings_utils import get_embedding
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, StringVariable
import httpx
import json
from pyfuseki import FusekiUpdate

class SubirOntologia(OWWidget):
    name = "SubirOntologia"
    description = "Subir a fuseki"
    icon = "icons/ontology.svg"
    priority = 10
    
    class Inputs:
        periodicos = Input("Periodicos", Table)
    
    def __init__(self):
        super().__init__()

    @Inputs.periodicos
    def set_periodicos(self, periodicos):
        if periodicos is not None:
            self.periodicos = periodicos
            self.runFuseki()
        else:
            self.periodicos = None
    
    def runFuseki(self):
        url = "http://localhost:3030"
        endpoint="dataset1"   
        fuseki_update = FusekiUpdate(url, endpoint)
        
        ids_periodicos=[]
        dic_periodicos={}
        contador_ids=1
        contador_pages=1
        contador_textos=1
        for row in self.periodicos:
            name = str(row["name"])
            texto = str(row["text"])
            entidades = str(row["entidades"])
            id2 = str(row["id"])
            id_page = str(row["idpage"])
            embending=str(row["embending"])
            generator=str(row["Generator"])
            viewport=str(row["viewport"])
            dateAccepted=str(row["DCTERMS.dateAccepted"])
            available=str(row["DCTERMS.available"])
            issued=str(row["DCTERMS.issued"])
            identifier=str(row["DC.identifier"])
            abstract=str(row["DCTERMS.abstract"])
            extent=str(row["DCTERMS.extent"])
            language=str(row["DC.language"])
            publisher=str(row["DC.publisher"])
            subject=str(row["DC.subject"])
            title=str(row["DC.title"])
            type1=str(row["DC.type"])
            citation_keywords=str(row["citation_keywords"])
            citation_title=str(row["citation_title"])
            citation_publisher=str(row["citation_publisher"])
            citation_language=str(row["citation_language"])
            citation_pdf_url=str(row["citation_pdf_url"])
            citation_date=str(row["citation_date"])
            citation_abstract_html_url=str(row["citation_abstract_html_url"])
            if id2 not in ids_periodicos:
                nombre_periodico="periodico"+str(contador_ids)
                nombre_citation="citacion"+str(contador_ids)
                ids_periodicos.append(id2)
                dic_periodicos.update({id2:nombre_periodico})
                contador_ids+=1
                sparql_str = "prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "+"prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> "+"prefix xsd: <http://www.w3.org/2001/XMLSchema#> "+"prefix sc: <https://schema.org/> "+"prefix ex: <http://newsont.com/> "+"insert data{ "+f"ex:{nombre_periodico} a sc:Newspaper; "+f"ex:generator \"{generator}\"; "+f"sc:size \"{viewport}\"; "+f"ex:dateAccepted \"{dateAccepted}\"^^xsd:date; "+f"sc:datePublished \"{available}\"^^xsd:date; "+f"sc:sdDatePublished \"{issued}\"^^xsd:date; "+f"sc:url \"{identifier}\"; "+f"sc:abstract \"{abstract}\"; "+f"sc:materialExtent \"{extent}\"; "+f"sc:inLanguage \"{language}\"; "+f"sc:version \"{publisher}\"; "+f"sc:about \"{subject}\"; "+f"sc:headline \"{title}\"; "+f"sc:keywords \"{citation_keywords}\". "+f"ex:{nombre_citation} a ex:Citation; "+f"sc:headline \"{citation_title}\"; "+f"sc:version \"{citation_publisher}\"; "+f"sc:inLanguage \"{citation_language}\"; "+f"sc:url \"{citation_pdf_url}\"; "+f"sc:sdDatePublished \"{citation_date}\"^^xsd:date; "+f"ex:citationAbstractUrl \"{citation_abstract_html_url}\". "+f"ex:{nombre_periodico} ex:citationProperty ex:{nombre_citation}. "+"}"
                query_result = fuseki_update.run_sparql(sparql_str)
        for row in self.periodicos:
            name = str(row["name"])
            texto = str(row["text"]).replace("\n","").replace("{","").replace("\"","").replace(":","").replace("}","").replace("[","").replace("]","").replace("-","").replace("'","").replace("*","").replace("+","").replace(";","").replace("=","").replace("/","").replace("_","").replace("\\","")
            entidades = str(row["entidades"])
            id2 = str(row["id"])
            id_page = str(row["idpage"])
            index = 1
            str1=str(row["embending"])
            output = str1[:index] + str1[index + 1: ]
            output=output[:-2]+"]"
            output=output.replace("\'","\"")
            json_object = json.loads(output)
            nombre_periodico=dic_periodicos[id2]
            nombre_page="pagina"+str(contador_pages)
            contador_pages+=1
            sparql_str = "prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "+"prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> "+"prefix xsd: <http://www.w3.org/2001/XMLSchema#> "+"prefix sc: <https://schema.org/> "+"prefix ex: <http://newsont.com/> "+"insert data{ "+f"ex:{nombre_page} a ex:Page. "+f"ex:{nombre_periodico} ex:hasPage ex:{nombre_page}. "+"}"
            query_result = fuseki_update.run_sparql(sparql_str)
            for entidad in entidades.split(","):
                if "http" in entidad:
                    sparql_str = "prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "+"prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> "+"prefix xsd: <http://www.w3.org/2001/XMLSchema#> "+"prefix sc: <https://schema.org/> "+"prefix ex: <http://newsont.com/> "+"insert data{ "+f"ex:{nombre_page} sc:mentions <{entidad}>. "+"}"
                    query_result = fuseki_update.run_sparql(sparql_str)
                else: 
                    ent=entidad.strip().replace("\n","").replace("{","").replace("\"","").replace(":","").replace("}","").replace("[","").replace("]","").replace("-","").replace("'","").replace("*","").replace("+","").replace(";","").replace("=","").replace("/","").replace("_","").replace("\\","")
                    sparql_str = "prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "+"prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> "+"prefix xsd: <http://www.w3.org/2001/XMLSchema#> "+"prefix sc: <https://schema.org/> "+"prefix ex: <http://newsont.com/> "+"insert data{ "+f"ex:{nombre_page} sc:mentions \"{ent}\". "+"}"
                    query_result = fuseki_update.run_sparql(sparql_str)
            for a in json_object[0]:
                nombre_texto="texto"+str(contador_textos)
                contador_textos+=1
                texto=a["text"]
                sparql_str = "prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "+"prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> "+"prefix xsd: <http://www.w3.org/2001/XMLSchema#> "+"prefix sc: <https://schema.org/> "+"prefix ex: <http://newsont.com/> "+"insert data{ "+f"ex:{nombre_texto} a ex:Texto; "+f"ex:text \"{texto}\". "+f"ex:{nombre_page} ex:hasText ex:{nombre_texto}. "+"}"
                query_result = fuseki_update.run_sparql(sparql_str)
                em=""
                for e in a["embedding"]:
                    em=em+str(e)+","
                em=em[:-1]
                sparql_str = "prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> "+"prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> "+"prefix xsd: <http://www.w3.org/2001/XMLSchema#> "+"prefix sc: <https://schema.org/> "+"prefix ex: <http://newsont.com/> "+"insert data{ "+f"ex:{nombre_texto} ex:embending \"{em}\". "+"}"
                query_result = fuseki_update.run_sparql(sparql_str)